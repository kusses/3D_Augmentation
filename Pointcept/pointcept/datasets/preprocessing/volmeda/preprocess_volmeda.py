import os
import sys
import argparse
import numpy as np
import pandas as pd
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import glob
import plyfile
from plyfile import PlyData, PlyElement
from scipy.spatial import KDTree

LABEL_TO_NAMES = {
    0: "Beanie", 1: "Cap", 2: "Sweatshirt", 3: "Shirt", 4: "Knitwear", 5: "Jeans",
    6: "JoggerPants", 7: "Pants", 8: "Slacks", 9: "Skirt", 10: "Loafers",
    11: "Sneakers", 12: "Shoes", 13: "Backpack", 14: "Sleeve", 15: "Boots", 16: "Mannequin"
}

def read_plymesh(filepath):
    """Read ply file and return it as numpy array. Returns None if empty."""
    try:
        with open(filepath, "rb") as f:
            plydata = plyfile.PlyData.read(f)
        if plydata.elements:
            vertices = pd.DataFrame(plydata["vertex"].data).values
            return vertices
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None, None

def vertex_normal(coords, k=10):
    kdtree = KDTree(coords)
    normals = np.zeros_like(coords)

    for i, coord in enumerate(coords):
        distances, indices = kdtree.query(coord, k=k)
        neighbors = coords[indices[1:]]  # 첫 번째 이웃은 자기 자신이므로 제외
        cov_matrix = np.cov(neighbors - coord, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov_matrix)
        normals[i] = eigvecs[:, 0]  # 최소 고유값에 대응하는 고유벡터가 노멀 벡터

    return normals

def map_label_to_index_volmeda(label):
    return LABEL_TO_NAMES.get(label, -1)  # Return -1 if label is not found

def process_labels(label):
    instance_ids = label.astype(np.int16)
    if np.any(np.isnan(instance_ids)) or not np.all(np.isfinite(instance_ids)):
        raise ValueError(f"Find NaN in instance IDs")
    print(f"Processed labels: {instance_ids}")  # Debug message
    return instance_ids

def scene_process(scene_id, split_name, mesh_path, output_path, parse_normals):
    print(f"Processing: {scene_id} in {split_name}")

    if os.path.exists(mesh_path):
        plydata = PlyData.read(mesh_path)

        vertices = np.vstack([plydata['vertex'][prop] for prop in ['x', 'y', 'z', 'red', 'green', 'blue']])
        vertices = vertices.transpose()
        coords = vertices[:, :3]
        colors = vertices[:, 3:6]

        save_dict = {
            'coord': coords.astype(np.float32),
            'color': colors.astype(np.uint8),
        }

        if parse_normals:
            save_dict["normal"] = vertex_normal(coords).astype(np.float32)

        if split_name != "test":
            if "class" in plydata["vertex"].data.dtype.names:
                point_labels = plydata["vertex"]["class"]
                print(f"point_labels: {point_labels}")  # Debug message
                save_dict["segment"] = process_labels(point_labels)
                print(f"save_dict['segment']: {save_dict['segment']}")  # Debug message
            else:
                print(f"Error: 'class' field not found in {mesh_path}")

        os.makedirs(output_path, exist_ok=True)

        for key, value in save_dict.items():
            np.save(os.path.join(output_path, f"{key}.npy"), value)
            loaded_value = np.load(os.path.join(output_path, f"{key}.npy"))  # Verify saved data
            print(f"Saved {key}: {loaded_value}")  # Debug message

    else:
        print(f"Error: File {mesh_path} not found")

def handle_process(scene_path, output_root, train_scenes, val_scenes, test_scenes, parse_normals=True):
    scene_id = os.path.basename(scene_path)
    mesh_path = scene_path

    try:
        if scene_id in train_scenes:
            output_path = os.path.join(output_root, "train", scene_id)
            split_name = "train"
            scene_process(scene_id, split_name, mesh_path, output_path, parse_normals)

        if scene_id in val_scenes:
            output_path = os.path.join(output_root, "val", scene_id)
            split_name = "val"
            scene_process(scene_id, split_name, mesh_path, output_path, parse_normals)

        if scene_id in test_scenes:
            output_path = os.path.join(output_root, "test", scene_id)
            split_name = "test"
            scene_process(scene_id, split_name, mesh_path, output_path, parse_normals)

    except Exception as e:
        print(f"Error processing {scene_id}: {e}")

if __name__ == "__main__":
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(ROOT_DIR)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True, help="Root directory of the dataset")
    parser.add_argument("--output_root", required=True, help="Output root directory")
    parser.add_argument("--parse_normals", default=True, type=bool, help="Whether to parse point normals")
    parser.add_argument("--num_workers", default=mp.cpu_count(), type=int, help="Number of workers for preprocessing")
    config = parser.parse_args()

    labels_pd = pd.DataFrame(list(LABEL_TO_NAMES.items()), columns=['id', 'raw_category'])
    with open(os.path.join(ROOT_DIR, "meta_data", "train.txt")) as f:
        train_scenes = f.read().splitlines()
    with open(os.path.join(ROOT_DIR, "meta_data", "val.txt")) as f:
        val_scenes = f.read().splitlines()
    with open(os.path.join(ROOT_DIR, "meta_data", "test.txt")) as f:
        test_scenes = f.read().splitlines()

    scene_paths = sorted(glob.glob(os.path.join(config.dataset_root, "volmeda", "*")))

    print("Processing scenes...")
    pool = ProcessPoolExecutor(max_workers=config.num_workers)
    _ = list(
        pool.map(
            handle_process,
            scene_paths,
            repeat(config.output_root),
            repeat(train_scenes),
            repeat(val_scenes),
            repeat(test_scenes),
            repeat(config.parse_normals),
        )
    )
