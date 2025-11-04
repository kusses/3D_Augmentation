import os
import sys
import argparse
import numpy as np
import glob
from shutil import copyfile
from scipy.spatial import KDTree

LABEL_TO_NAMES = {
    0: "Beanie", 1: "Cap", 2: "Sweatshirt", 3: "Shirt", 4: "Knitwear", 5: "Jeans",
    6: "JoggerPants", 7: "Pants", 8: "Slacks", 9: "Skirt", 10: "Loafers",
    11: "Sneakers", 12: "Shoes", 13: "Backpack", 14: "Sleeve", 15: "Boots", 16: "Mannequin"
}

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

def copy_files(scene_id, split_name, output_root, dataset_root):
    """Copy numpy files to the appropriate train, val, or test folders."""

    if not scene_id.endswith(".ply"):
        scene_id_with_ply = scene_id + ".ply"
    else:
        scene_id_with_ply = scene_id

    source_folder = os.path.join(dataset_root, scene_id)
    output_path = os.path.join(output_root, split_name, scene_id_with_ply)  # .ply added to the folder name
    os.makedirs(output_path, exist_ok=True)

    for npy_file in glob.glob(os.path.join(source_folder, "*.npy")):
        try:
            destination = os.path.join(output_path, os.path.basename(npy_file))
            copyfile(npy_file, destination)
        except Exception as e:
            print(f"Error copying {npy_file} to {destination}: {e}")

    # Check for the existence of normal.npy
    normal_file = os.path.join(output_path, "normal.npy")


    if not os.path.exists(normal_file):
        # Look for any file with 'coord' in its name
        coord_files = [f for f in glob.glob(os.path.join(output_path, "*coord*.npy"))]
        if coord_files:
            for coord_file in coord_files:
                try:
                    coords = np.load(coord_file, allow_pickle=True)
                    print(f"Loaded coords from {coord_file}: {coords.shape}")  # Debugging statement

                    # Check if the coords is a valid 2D array with 3 columns (x, y, z)
                    if isinstance(coords, np.ndarray) and coords.ndim == 2 and coords.shape[1] == 3:
                        normals = vertex_normal(coords)
                        np.save(normal_file, normals)
                        print(f"Generated and saved normals to {normal_file}")
                        break  # Stop after processing the first valid coord file
                    else:
                        print(f"Unexpected content in {coord_file}. Expected a 2D array with 3 columns.")
                except Exception as e:
                    print(f"Error processing {coord_file}: {e}")
        else:
            print(f"No suitable coord file found in {output_path}.")

def handle_process(scene_id, output_root, train_scenes, val_scenes, test_scenes, dataset_root):
    """Determine the split and copy files accordingly."""
    if scene_id in train_scenes:
        copy_files(scene_id, "train", output_root, dataset_root)

    if scene_id in val_scenes:
        copy_files(scene_id, "val", output_root, dataset_root)

    if scene_id in test_scenes:
        copy_files(scene_id, "test", output_root, dataset_root)

if __name__ == "__main__":
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(ROOT_DIR)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True, help="Root directory of the dataset")
    parser.add_argument("--output_root", required=True, help="Output root directory")
    config = parser.parse_args()

    # Load scene IDs from the text files
    with open(os.path.join(ROOT_DIR, "meta_data", "train.txt")) as f:
        train_scenes = f.read().splitlines()
    with open(os.path.join(ROOT_DIR, "meta_data", "val.txt")) as f:
        val_scenes = f.read().splitlines()
    with open(os.path.join(ROOT_DIR, "meta_data", "test.txt")) as f:
        test_scenes = f.read().splitlines()

    # Collect all scene folders from the dataset root
    scene_folders = [os.path.basename(folder) for folder in sorted(glob.glob(os.path.join(config.dataset_root, "*")))]

    # Process each scene folder
    print("Processing scenes...")
    for scene_id in scene_folders:
        handle_process(scene_id, config.output_root, train_scenes, val_scenes, test_scenes, config.dataset_root)
