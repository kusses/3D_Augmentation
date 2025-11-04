"""
Utils for Datasets

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""
import random
from collections.abc import Mapping, Sequence
import numpy as np
import torch
from torch.utils.data.dataloader import default_collate

'''
def collate_fn(batch):
    """
    collate function for point cloud which support dict and list,
    'coord' is necessary to determine 'offset'
    """
    if not isinstance(batch, Sequence):
        raise TypeError(f"{batch.dtype} is not supported.")

    if isinstance(batch[0], torch.Tensor):
        return torch.cat(list(batch))
    elif isinstance(batch[0], str):
        # str is also a kind of Sequence, judgement should before Sequence
        return list(batch)
    elif isinstance(batch[0], Sequence):
        for data in batch:
            data.append(torch.tensor([data[0].shape[0]]))
        batch = [collate_fn(samples) for samples in zip(*batch)]
        batch[-1] = torch.cumsum(batch[-1], dim=0).int()
        return batch
    elif isinstance(batch[0], Mapping):
        batch = {key: collate_fn([d[key] for d in batch]) for key in batch[0]}
        for key in batch.keys():
            if "offset" in key:
                batch[key] = torch.cumsum(batch[key], dim=0)
        return batch
    else:
        return default_collate(batch)
'''
def collate_fn(batch):
    """
    Collate function for point clouds supporting dict and list.
    'coord' is necessary to determine 'offset'.
    """
    if not isinstance(batch, Sequence):
        raise TypeError(f"{type(batch)} is not supported.")

    if isinstance(batch[0], torch.Tensor):
        return torch.cat(list(batch))
    elif isinstance(batch[0], str):
        return list(batch)
    elif isinstance(batch[0], Sequence):
        # Handle list of tensors
        if isinstance(batch[0][0], torch.Tensor):
            lengths = [item[0].shape[0] for item in batch]
            for i, data in enumerate(batch):
                data.append(torch.tensor([lengths[i]]))
            batch = [collate_fn(samples) for samples in zip(*batch)]
            batch[-1] = torch.cumsum(batch[-1], dim=0).int()
            return batch
        else:
            return [collate_fn(samples) for samples in zip(*batch)]
    elif isinstance(batch[0], Mapping):
        batch = {key: collate_fn([d[key] for d in batch]) for key in batch[0]}
        for key in batch.keys():
            if "offset" in key:
                batch[key] = torch.cumsum(batch[key], dim=0)
        return batch
    else:
        return default_collate(batch)


def point_collate_fn(batch, mix_prob=0):
    assert isinstance(
        batch[0], Mapping
    )  # currently, only support input_dict, rather than input_list

    batch = collate_fn(batch)

    if "offset" in batch.keys():
        # Mix3d (https://arxiv.org/pdf/2110.02210.pdf)
        if random.random() < mix_prob:
            batch["offset"] = torch.cat(
                [batch["offset"][1:-1:2], batch["offset"][-1].unsqueeze(0)], dim=0
            )

def point_collate_fn(batch, mix_prob=0):
    assert isinstance(
        batch[0], Mapping
    )  # currently, only support input_dict, rather than input_list
    batch = collate_fn(batch)
    if "offset" in batch.keys():
        # Mix3d (https://arxiv.org/pdf/2110.02210.pdf)
        if random.random() < mix_prob:
            batch["offset"] = torch.cat(
                [batch["offset"][1:-1:2], batch["offset"][-1].unsqueeze(0)], dim=0
            )
    return batch

    # (JY) for inference, make 'batch' key


def point_collate_fn_infer(batch):
    assert isinstance(batch[0], Mapping)

    fragment_list = batch[0]["fragment_list"]

    # Convert numpy -> tensor
    for frag in fragment_list:
        for key, val in frag.items():
            if isinstance(val, np.ndarray):
                frag[key] = torch.from_numpy(val)

    required_keys = {"coord", "feat", "offset", "index"}
    if not all(required_keys.issubset(frag.keys()) for frag in fragment_list):
        raise ValueError("Missing keys in fragment_list")

    point_batch = point_collate_fn(fragment_list)

    # name
    point_batch["name"] = batch[0]["name"]

    # concat segment across fragments
    segment_list = []
    for frag in fragment_list:
        seg = frag.get("segment")
        if seg is not None:
            if isinstance(seg, np.ndarray):
                seg = torch.from_numpy(seg).long()
            segment_list.append(seg)


    return point_batch


def gaussian_kernel(dist2: np.array, a: float = 1, c: float = 5):
    return a * np.exp(-dist2 / (2 * c**2))