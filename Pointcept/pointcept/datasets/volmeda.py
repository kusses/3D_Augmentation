import os
import numpy as np
import torch
from torch.utils.data import Dataset
from copy import deepcopy

from pointcept.datasets.builder import DATASETS
from pointcept.datasets.defaults import DefaultDataset
from pointcept.utils.cache import shared_dict


@DATASETS.register_module()
class volmeda(DefaultDataset):
    VALID_ASSETS = [
        "coord",
        "color",
        "normal",
        "segment"
    ]
    LABEL_TO_NAMES = {
        0: 'Beanie', 1: 'Cap', 2: 'Sweatshirt', 3: 'Shirt', 4: 'Knitwear', 5: 'Jeans',
        6: 'JoggerPants', 7: 'Pants', 8: 'Slacks', 9: 'Skirt', 10: 'Loafers',
        11: 'Sneakers', 12: 'Shoes', 13: 'Backpack', 14: 'Sleeve', 15: 'Boots', 16: 'Mannequin'
    }

    def __init__(
        self,
        lr_file=None,
        la_file=None,
        **kwargs,
    ):
        self.lr = np.loadtxt(lr_file, dtype=str) if lr_file is not None else None
        self.la = torch.load(la_file) if la_file is not None else None
        super().__init__(**kwargs)

    def get_data_list(self):
        if self.lr is None:
            data_list = super().get_data_list()
        else:
            data_list = [
                os.path.join(self.data_root, "train", name) for name in self.lr
            ]
        return data_list

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        name = self.get_data_name(idx)

        # for the case of cache loading
        if self.cache:
            cache_name = f"pointcept-{name}"
            return shared_dict(cache_name)

        data_dict = {}
        assets = os.listdir(data_path)


        # (coord, color, normal, segment) initialization
        expected_assets = {"coord": False, "color": False, "normal": False, "segment": False}

        for asset in assets:
            if not asset.endswith(".npy"):
                continue

            asset_name = asset.rsplit('.', 1)[0]

            # .ply delete
            if asset_name.endswith(".ply"):
                asset_name = asset_name[:-4]

            #
            if 'coord' in asset_name:
                asset_name = 'coord'
            elif 'color' in asset_name:
                asset_name = 'color'
            elif 'segment' in asset_name:
                asset_name = 'segment'
            elif 'normal' in asset_name:
                asset_name = 'normal'

            # processing VALID_ASSETS
            if asset_name in self.VALID_ASSETS:
                try:
                    data_dict[asset_name] = np.load(os.path.join(data_path, asset))
                    expected_assets[asset_name] = True  # VALID ASSET loading success
                except Exception as e:
                    print(f"Error loading {asset_name} from {data_path}: {e}")

        # Check for invalid label-only sample (all labels are ignore_index)
        if "segment" in data_dict:
            segment = data_dict["segment"]
            if (segment == -1).all():
                # Return dummy data + invalid_sample flag
                dummy_tensor = np.zeros((1, 3), dtype=np.float32)
                data_dict = {
                    "coord": dummy_tensor,
                    "color": dummy_tensor,
                    "normal": dummy_tensor,
                    "segment": np.array([-1], dtype=np.int32),
                    "name": name,
                    "invalid_sample": True
                }

        # VALID_ASSET process
        data_dict["name"] = name
        data_dict["coord"] = data_dict["coord"].astype(np.float32)
        data_dict["color"] = data_dict["color"].astype(np.float32)
        data_dict["normal"] = data_dict["normal"].astype(np.float32)
        if "segment" in data_dict.keys():
            data_dict["segment"] = data_dict["segment"].reshape([-1]).astype(np.int32)
        else:
            data_dict["segment"] = (
                    np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
            )

        return data_dict

