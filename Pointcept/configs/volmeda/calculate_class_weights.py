import os
import sys
import numpy as np
import torch

# Add the parent directory of `pointcept` to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'pointcept')))

# Import the volmeda class
from pointcept.datasets.volmeda import volmeda

def calculate_class_weights(data_root, split, num_classes, ignore_index=-1):
    dataset = volmeda(
        data_root=data_root,
        split=split,
        lr_file=None,  # Adjust this based on your needs
        la_file=None   # Adjust this based on your needs
    )
    class_counts = {i: 0 for i in range(num_classes)}
    for idx in range(len(dataset.data_list)):
        data = dataset.get_data(idx)
        segments = data["segment"]
        unique, counts = np.unique(segments, return_counts=True)
        for u, c in zip(unique, counts):
            if u != ignore_index:
                class_counts[u] += c

    total_samples = sum(class_counts.values())
    class_weights = []
    for i in range(num_classes):
        if class_counts[i] == 0:
            class_weights.append(float(10.0))
        else:
            class_weights.append(total_samples / (num_classes * class_counts[i]))

    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    return class_weights_tensor

if __name__ == "__main__":
    #data_root = "/data/volme/data/pointcept_data/volmeda_pointcept"
    #data_root = "/data/volme/data/pointcept_data/volmeda_pointcept_aug"
    data_root = "/data/volme/data/VOLMEDATA/volmeda"
    num_classes = 17
    ignore_index = -1
    weights = calculate_class_weights(data_root, "train", num_classes, ignore_index)
    print(weights)
    torch.save(weights, "class_weights_volmedata.pth")
