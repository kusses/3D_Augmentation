"""
Trainer

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import sys
import weakref
import torch
import torch.nn as nn
import torch.utils.data
from torch.nn.utils import clip_grad_norm_
from functools import partial
import time


if sys.version_info >= (3, 10):
    from collections.abc import Iterator
else:
    from collections import Iterator
from tensorboardX import SummaryWriter

from .defaults import create_ddp_model, worker_init_fn
from .hooks import HookBase, build_hooks
import pointcept.utils.comm as comm
from pointcept.datasets import build_dataset, point_collate_fn, collate_fn
from pointcept.models import build_model
from pointcept.utils.logger import get_root_logger
from pointcept.utils.optimizer import build_optimizer
from pointcept.utils.scheduler import build_scheduler
from pointcept.utils.events import EventStorage
from pointcept.utils.registry import Registry
from pointcept.utils.hash_utils import hash_file, hash_directory
import os
import csv
from datetime import datetime


TRAINERS = Registry("trainers")


class TrainerBase:
    def __init__(self) -> None:
        self.hooks = []
        self.epoch = 0
        self.start_epoch = 0
        self.max_epoch = 0
        self.max_iter = 0
        self.comm_info = dict()
        self.data_iterator: Iterator = enumerate([])
        self.storage: EventStorage
        self.writer: SummaryWriter

    def register_hooks(self, hooks) -> None:
        hooks = build_hooks(hooks)
        for h in hooks:
            assert isinstance(h, HookBase)
            # To avoid circular reference, hooks and trainer cannot own each other.
            # This normally does not matter, but will cause memory leak if the
            # involved objects contain __del__:
            # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
            h.trainer = weakref.proxy(self)
        self.hooks.extend(hooks)

    def train(self):

        with EventStorage() as self.storage:
            # => before train
            self.before_train()
            for self.epoch in range(self.start_epoch, self.max_epoch):
                # => before epoch
                self.before_epoch()
                # => run_epoch
                for (
                    self.comm_info["iter"],
                    self.comm_info["input_dict"],
                ) in self.data_iterator:
                    # => before_step
                    self.before_step()
                    # => run_step
                    self.run_step()
                    # => after_step
                    self.after_step()
                # => after epoch
                self.after_epoch()
            # => after train
            self.after_train()

    def before_train(self):
        for h in self.hooks:
            h.before_train()

    def before_epoch(self):
        for h in self.hooks:
            h.before_epoch()

    def before_step(self):
        for h in self.hooks:
            h.before_step()

    def run_step(self):
        raise NotImplementedError

    def after_step(self):
        for h in self.hooks:
            h.after_step()

    def after_epoch(self):
        for h in self.hooks:
            h.after_epoch()
        self.storage.reset_histories()

    def after_train(self):
        # Sync GPU before running train hooks
        comm.synchronize()
        for h in self.hooks:
            h.after_train()
        if comm.is_main_process():
            self.writer.close()



@TRAINERS.register_module("DefaultTrainer")
class Trainer(TrainerBase):
    def __init__(self, cfg):
        super(Trainer, self).__init__()
        self.epoch = 0
        self.start_epoch = 0
        self.max_epoch = cfg.eval_epoch
        self.best_metric_value = -torch.inf
        self.logger = get_root_logger(
            log_file=os.path.join(cfg.save_path, "train.log"),
            file_mode="a" if cfg.resume else "w",
        )
        start_time = datetime.now()
        self.logger.info(f"[TIME] Training started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("=> Loading config ...")
        self.cfg = cfg
        self.logger.info(f"Save path: {cfg.save_path}")
        self.logger.info(f"Config:\n{cfg.pretty_text}")
        self.logger.info("=> Building model ...")
        self.model = self.build_model()
        self.logger.info("=> Building writer ...")
        self.writer = self.build_writer()
        self.logger.info("=> Building train dataset & dataloader ...")
        self.train_loader = self.build_train_loader()
        self.logger.info("=> Building val dataset & dataloader ...")
        self.val_loader = self.build_val_loader()
        self.logger.info("=> Building optimize, scheduler, scaler(amp) ...")
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()
        self.scaler = self.build_scaler()
        self.logger.info("=> Building hooks ...")
        self.register_hooks(self.cfg.hooks)

    def train(self):
        # Save hash
        if comm.is_main_process():
            try:
                # 경로 정의
                dataset_dir = self.cfg.data.train.data_root
                model_weight_path = self.cfg.weight if hasattr(self.cfg, "weight") and self.cfg.weight else None
                best_model_path = os.path.join(self.cfg.save_path, "best.pth")
                code_root = os.path.join(os.getcwd(), "pointcept")

                # 해시 계산
                data_hash = hash_directory(dataset_dir, exts=[".ply", ".npz", ".npy", ".txt"])
                model_hash = hash_file(model_weight_path) if model_weight_path and os.path.exists(
                    model_weight_path) else "NO_WEIGHT"
                dataset_code_hash = hash_directory(os.path.join(code_root, "datasets"), exts=[".py"])
                model_code_hash = hash_directory(os.path.join(code_root, "models"), exts=[".py"])
                engine_code_hash = hash_directory(os.path.join(code_root, "engines"), exts=[".py"])
                total_code_hash = hash_directory(code_root, exts=[".py"])

                # 모델 저장 시간 확인
                if os.path.exists(best_model_path):
                    best_model_time = datetime.fromtimestamp(os.path.getmtime(best_model_path)).isoformat()
                else:
                    best_model_time = "N/A"

                # 데이터 개수 및 형태 확인
                try:
                    sample = self.train_loader.dataset[0]
                    num_samples = len(self.train_loader.dataset)
                    if isinstance(sample, dict) and "coord" in sample:
                        num_points = sample["coord"].shape[0]
                        feature_dim = sample["coord"].shape[1]
                    elif isinstance(sample, (tuple, list)) and isinstance(sample[0], dict):
                        d = sample[0]
                        num_points = d["coord"].shape[0]
                        feature_dim = d["coord"].shape[1]
                    else:
                        num_points, feature_dim = "?", "?"
                except Exception as e:
                    self.logger.warning(f"[HASH] Failed to extract data shape info: {e}")
                    num_samples, num_points, feature_dim = "?", "?", "?"

                # CSV 저장
                csv_path = os.path.join(self.cfg.save_path, "hash_log.csv")
                with open(csv_path, mode="w", newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([
                        "Timestamp", "DataHash", "InitialWeightHash",
                        "DatasetCodeHash", "ModelCodeHash", "EngineCodeHash", "TotalCodeHash",
                        "BestModelSavedTime", "NumSamples", "NumPoints", "FeatureDim"
                    ])
                    writer.writerow([
                        datetime.now().isoformat(),
                        data_hash,
                        model_hash,
                        dataset_code_hash,
                        model_code_hash,
                        engine_code_hash,
                        total_code_hash,
                        best_model_time,
                        num_samples,
                        num_points,
                        feature_dim
                    ])
                self.logger.info(f"[HASH] Hash log saved to {csv_path}")

            except Exception as e:
                self.logger.warning(f"[HASH] Failed to compute/save hash values: {e}")

        with EventStorage() as self.storage:
            # => before train
            self.before_train()
            self.logger.info(">>>>>>>>>>>>>>>> Start Training >>>>>>>>>>>>>>>>")
            for self.epoch in range(self.start_epoch, self.max_epoch):
                epoch_start_time = datetime.now()
                self.logger.info(
                    f"[TIME] >>> Epoch {self.epoch + 1}/{self.max_epoch} started at {epoch_start_time.strftime('%H:%M:%S')}")

                # => before epoch
                # TODO: optimize to iteration based
                if comm.get_world_size() > 1:
                    self.train_loader.sampler.set_epoch(self.epoch)
                self.model.train()
                self.data_iterator = enumerate(self.train_loader)
                self.before_epoch()
                # => run_epoch
                for (
                    self.comm_info["iter"],
                    self.comm_info["input_dict"],
                ) in self.data_iterator:
                    # => before_step
                    self.before_step()
                    # => run_step
                    self.run_step()
                    # => after_step
                    self.after_step()
                # => after epoch
                self.after_epoch()
                # epoch end time
                elapsed = datetime.now() - epoch_start_time
                self.logger.info(
                    f"[TIME] <<< Epoch {self.epoch + 1} ended at {datetime.now().strftime('%H:%M:%S')} (Duration: {elapsed.total_seconds():.2f} seconds)")

                # (JY) after each epoch clear GPU memory
                if self.cfg.empty_cache_per_epoch:
                    torch.cuda.empty_cache()
            # => after train
            self.after_train()

            #hash save
            if comm.is_main_process():
                try:
                    dataset_dir = self.cfg.data.train.data_root
                    model_weight_path = self.cfg.weight if hasattr(self.cfg, "weight") else "N/A"
                    code_root = os.path.join(os.getcwd(), "pointcept")

                    data_hash = hash_directory(dataset_dir, exts=[".ply", ".npz", ".npy", ".txt"])
                    model_hash = hash_file(model_weight_path) if os.path.exists(model_weight_path) else "NO_WEIGHT"
                    dataset_code_hash = hash_directory(os.path.join(code_root, "datasets"), exts=[".py"])
                    model_code_hash = hash_directory(os.path.join(code_root, "models"), exts=[".py"])
                    engine_code_hash = hash_directory(os.path.join(code_root, "engines"), exts=[".py"])
                    total_code_hash = hash_directory(code_root, exts=[".py"])

                    if isinstance(model_weight_path, str) and os.path.exists(model_weight_path):
                        model_mtime = datetime.fromtimestamp(os.path.getmtime(model_weight_path)).isoformat()
                    else:
                        model_mtime = "N/A"

                    try:
                        sample = self.train_loader.dataset[0]
                        num_samples = len(self.train_loader.dataset)
                        if isinstance(sample, dict) and "coord" in sample:
                            num_points = sample["coord"].shape[0]
                            feature_dim = sample["coord"].shape[1]
                        elif isinstance(sample, (tuple, list)) and isinstance(sample[0], dict):
                            d = sample[0]
                            num_points = d["coord"].shape[0]
                            feature_dim = d["coord"].shape[1]
                        else:
                            num_points, feature_dim = "?", "?"
                    except Exception as e:
                        self.logger.warning(f"[HASH] Failed to extract data shape info: {e}")
                        num_samples, num_points, feature_dim = "?", "?", "?"

                    csv_path = os.path.join(self.cfg.save_path, "hash_log_final.csv")
                    with open(csv_path, mode="w", newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([
                            "Timestamp", "DataHash", "ModelWeightHash",
                            "DatasetCodeHash", "ModelCodeHash", "EngineCodeHash", "TotalCodeHash",
                            "ModelSavedTime", "NumSamples", "NumPoints", "FeatureDim"
                        ])
                        writer.writerow([
                            datetime.now().isoformat(),
                            data_hash,
                            model_hash,
                            dataset_code_hash,
                            model_code_hash,
                            engine_code_hash,
                            total_code_hash,
                            model_mtime,
                            num_samples,
                            num_points,
                            feature_dim
                        ])
                    self.logger.info(f"[HASH] Final hash log saved to {csv_path}")
                except Exception as e:
                    self.logger.warning(f"[HASH] Failed to compute/save final hash values: {e}")
    def run_step(self):
        input_dict = self.comm_info["input_dict"]
        # (JY) inspect NaN / Inf
        for key, val in input_dict.items():
            if isinstance(val, torch.Tensor):
                if torch.isnan(val).any() or torch.isinf(val).any():
                    self.logger.error(
                        f"[NaN/Inf DETECTED] {key} contains invalid values at iter {self.comm_info['iter']}")
                    raise ValueError(f"Invalid value in input: {key}")

        if input_dict.get("invalid_sample", False):
            self.logger.warning(f"[SKIP] Invalid sample at iter {self.comm_info['iter']}")
            return
        # (JY) Invalid sample check (from dataset)
        if input_dict.get("invalid_sample", False):
            self.logger.warning(f"[SKIP] Invalid sample at iter {self.comm_info['iter']}")
            return

        for key in input_dict.keys():
            if isinstance(input_dict[key], torch.Tensor):
                input_dict[key] = input_dict[key].cuda(non_blocking=True)
        with torch.cuda.amp.autocast(enabled=self.cfg.enable_amp):
            output_dict = self.model(input_dict)
            loss = output_dict["loss"]


        if torch.isnan(loss).any() or torch.isinf(loss).any():
            self.logger.error("[DEBUG] Loss computation failed.")
            self.logger.error(f"[DEBUG] Output dict: {output_dict}")
            raise ValueError("Invalid loss value detected.")

        if torch.isnan(loss) or torch.isinf(loss):
            self.logger.error(f"[DEBUG] Invalid loss detected: {loss}")
            self.logger.error(f"[DEBUG] Input dict: {input_dict}")
            raise ValueError("Loss value is NaN or Inf. Check your data or model output.")
        else:
            #self.logger.info(f"[DEBUG] Loss at iteration {self.comm_info['iter']}: {loss.item()}")
            # (jy) Tensorboad monitoring
            if self.writer is not None:
                global_step = self.epoch * len(self.train_loader) + self.comm_info["iter"]
                self.writer.add_scalar("Train/Loss", loss.item(), self.epoch)
                #self.writer.add_scalar("Train/Loss", loss.item(), global_step)

                lr = self.optimizer.param_groups[0]["lr"]
                self.writer.add_scalar("Train/LR", lr, global_step)

        ## EOL debugging

        self.optimizer.zero_grad()
        if self.cfg.enable_amp:
            self.scaler.scale(loss).backward()

            # (JY) Add Gradient Clipping
            if self.cfg.grad_clip_norm is not None:
                self.scaler.unscale_(self.optimizer)  # clip after unscale
                clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)

            self.scaler.step(self.optimizer)

            # When enable amp, optimizer.step call are skipped if the loss scaling factor is too large.
            # Fix torch warning scheduler step before optimizer step.
            #scaler = self.scaler.get_scale()
            #self.scaler.update()
            #if scaler <= self.scaler.get_scale():
            #    self.scheduler.step()
            # (JY) AMP scaling state tracking
            scale_now = self.scaler.get_scale()
            self.scaler.update()
            scale_after = self.scaler.get_scale()
            if scale_now > scale_after:
                self.logger.warning(
                    f"[AMP WARNING] GradScaler skipped optimizer step due to inf/nan at iter {self.comm_info['iter']}")
            else:
                self.scheduler.step()
        else:
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
        if self.cfg.empty_cache:
            torch.cuda.empty_cache()
        self.comm_info["model_output_dict"] = output_dict

    def after_epoch(self):
        for h in self.hooks:
            h.after_epoch()
        self.storage.reset_histories()
        if self.cfg.empty_cache_per_epoch:
            torch.cuda.empty_cache()

    def build_model(self):
        model = build_model(self.cfg.model)
        if self.cfg.sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # logger.info(f"Model: \n{self.model}")
        self.logger.info(f"Num params: {n_parameters}")
        model = create_ddp_model(
            model.cuda(),
            broadcast_buffers=False,
            find_unused_parameters=self.cfg.find_unused_parameters,
        )
        return model

    def build_writer(self):
        writer = SummaryWriter(self.cfg.save_path) if comm.is_main_process() else None
        self.logger.info(f"Tensorboard writer logging dir: {self.cfg.save_path}")
        return writer

    def build_train_loader(self):
        train_data = build_dataset(self.cfg.data.train)

        if comm.get_world_size() > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        else:
            train_sampler = None

        init_fn = (
            partial(
                worker_init_fn,
                num_workers=self.cfg.num_worker_per_gpu,
                rank=comm.get_rank(),
                seed=self.cfg.seed,
            )
            if self.cfg.seed is not None
            else None
        )

        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.cfg.batch_size_per_gpu,
            shuffle=(train_sampler is None),
            num_workers=self.cfg.num_worker_per_gpu,
            sampler=train_sampler,
            collate_fn=partial(point_collate_fn, mix_prob=self.cfg.mix_prob),
            pin_memory=True,
            worker_init_fn=init_fn,
            drop_last=True,
            persistent_workers=True,
        )
        return train_loader

    def build_val_loader(self):
        val_loader = None
        if self.cfg.evaluate:
            val_data = build_dataset(self.cfg.data.val)
            if comm.get_world_size() > 1:
                val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
            else:
                val_sampler = None
            val_loader = torch.utils.data.DataLoader(
                val_data,
                batch_size=self.cfg.batch_size_val_per_gpu,
                shuffle=False,
                num_workers=self.cfg.num_worker_per_gpu,
                pin_memory=True,
                sampler=val_sampler,
                collate_fn=collate_fn,
            )
        return val_loader

    def build_optimizer(self):
        return build_optimizer(self.cfg.optimizer, self.model, self.cfg.param_dicts)

    def build_scheduler(self):
        assert hasattr(self, "optimizer")
        assert hasattr(self, "train_loader")
        self.cfg.scheduler.total_steps = len(self.train_loader) * self.cfg.eval_epoch
        return build_scheduler(self.cfg.scheduler, self.optimizer)

    def build_scaler(self):
        scaler = torch.cuda.amp.GradScaler() if self.cfg.enable_amp else None
        return scaler


@TRAINERS.register_module("MultiDatasetTrainer")
class MultiDatasetTrainer(Trainer):
    def build_train_loader(self):
        from pointcept.datasets import MultiDatasetDataloader

        train_data = build_dataset(self.cfg.data.train)
        train_loader = MultiDatasetDataloader(
            train_data,
            self.cfg.batch_size_per_gpu,
            self.cfg.num_worker_per_gpu,
            self.cfg.mix_prob,
            self.cfg.seed,
        )
        self.comm_info["iter_per_epoch"] = len(train_loader)
        return train_loader
