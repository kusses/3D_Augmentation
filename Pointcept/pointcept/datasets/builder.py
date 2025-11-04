"""
Dataset Builder

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from pointcept.utils.registry import Registry

DATASETS = Registry("datasets")


def build_dataset(cfg):
    """Build datasets."""
    print(f"data split - {cfg.get('split', 'Unknown Split')}")
    print(f"data root path: {cfg.get('data_root', 'Unknown Root')}")
    print(f"dataset type: {cfg.get('type', 'Unknown Type')}")
    print(f"Test mode: {cfg.get('test_mode', False)}")

    return DATASETS.build(cfg)