"""
Main Training Script

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from pointcept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from pointcept.engines.train import TRAINERS
from pointcept.engines.launch import launch
import torch
import time
import os


def main_worker(cfg):
    cfg = default_setup(cfg)
    # clear cache (JY)
    torch.cuda.empty_cache()
    trainer = TRAINERS.build(dict(type=cfg.train.type, cfg=cfg))
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    with open(os.path.join(cfg.save_path, "train_time.txt"), "w") as f:
        f.write(f"Total training time: {(end_time - start_time) / 60:.2f} minutes\n")


def main():
    args = default_argument_parser().parse_args()
    cfg = default_config_parser(args.config_file, args.options)

    launch(
        main_worker,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        cfg=(cfg,),
    )


if __name__ == "__main__":
    main()