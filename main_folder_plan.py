# This is agumentation code according to augmentation plan
# 2024.10.04 Jiyoun Lim ,ETRI

import os
import argparse
import csv
from SMOTE3d import SMOTE3D
from ColorChange import ColorChange
from visualization3d import visualize_ply, numpy_to_ply

def process_folder(in_data_folder, out_data_folder, subfolder, mode, augment_count):
    # Base output directory for this subfolder
    base_out_data_folder = os.path.join(out_data_folder, subfolder[:-4])  # Removing ".ply" from subfolder name

    # Create the base output folder if it doesn't exist
    if not os.path.exists(base_out_data_folder):
        os.makedirs(base_out_data_folder)
    for i in range(augment_count):
        identifier = f"_{i}"
        specific_out_folder = os.path.join(base_out_data_folder, f"aug_{i}")

        # Create specific output folder if it doesn't exist
        if not os.path.exists(specific_out_folder):
            os.makedirs(specific_out_folder)

        if mode == "color":
            change_color = ColorChange(in_data_folder, specific_out_folder, exclude_class=16)
            change_color.change_color_class()

        elif mode == "both":
            smote = SMOTE3D(in_data_folder, specific_out_folder, exclude_class=16)
            smote.augment_minority_class(k_neighbors=5)

        elif mode == "coord":
            smote = SMOTE3D(in_data_folder, specific_out_folder, exclude_class=16, apply_color_augmentation=False)
            smote.augment_minority_class(k_neighbors=5)

        # Set the path to the PLY files with identifiers
        original_ply_file = os.path.join(specific_out_folder, 'original_data.ply')
        augmented_ply_file = os.path.join(specific_out_folder, f'augmented_data{identifier}.ply')

        # Original Numpy file paths
        coord_file = os.path.join(in_data_folder, "coord.npy")
        color_file = os.path.join(in_data_folder, "color.npy")
        segment_file = os.path.join(in_data_folder, "segment.npy")

        # Check if input numpy files exist
        if not (os.path.exists(coord_file) and os.path.exists(color_file) and os.path.exists(segment_file)):
            print(f"Error: One of the input files does not exist in {in_data_folder}")
            continue

        # Convert numpy to PLY with identifiers
        numpy_to_ply(original_ply_file, coord_file, color_file, segment_file)

        # Augmented Numpy file paths
        aug_coord_file = os.path.join(specific_out_folder, f"augmented_coord{identifier}.npy")
        aug_color_file = os.path.join(specific_out_folder, f"augmented_color{identifier}.npy")
        aug_segment_file = os.path.join(specific_out_folder, f"augmented_segment{identifier}.npy")

        # Check if augmented numpy files exist before converting
        if not (os.path.exists(aug_coord_file) and os.path.exists(aug_color_file) and os.path.exists(aug_segment_file)):
            print(f"Error: Augmented numpy files not found in {specific_out_folder}")
            continue

        # Convert augmented numpy files to PLY
        numpy_to_ply(augmented_ply_file, aug_coord_file, aug_color_file, aug_segment_file)

        # Optionally visualize the original and augmented data
        # visualize_ply(original_ply_file)
        # visualize_ply(augmented_ply_file)

def load_augmentation_counts(csv_file):
    augment_counts = {}
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            fullid = row['fullid']
            count = int(row['augmentation_count'])
            augment_counts[fullid] = count
    return augment_counts

def main():
    # Set default paths
    default_in_data_folder = "/data/volme/data/pointcept_data/volmeda_pointcept/original"
    default_out_data_folder = "/data/volme/data/pointcept_data/volmeda_pointcept_all/volmeda_aug3"
    default_csv_file = "./data_prepare/fullid_augmentation_plan.csv"  # Update this with your CSV file path

    parser = argparse.ArgumentParser(description="Visualize and augment 3D point clouds.")
    parser.add_argument("--in_data_folder", type=str, default=default_in_data_folder,
                        help="Path to the folder containing subfolders with the original data.")
    parser.add_argument("--out_data_folder", type=str, default=default_out_data_folder,
                        help="Path to the folder to save the augmented data.")
    parser.add_argument("--mode", type=str, choices=["color", "both", "coord"], required=True,
                        help="Choose 'color' to change color only, or 'both' to change both color and coordinates.")
    parser.add_argument("--aug_plan_file", type=str, default=default_csv_file,
                        help="Path to the CSV file containing augmentation counts.")

    args = parser.parse_args()

    # Load augmentation counts from CSV
    augment_counts = load_augmentation_counts(args.aug_plan_file)

    # Iterate through all subdirectories in the input data folder
    for subfolder in os.listdir(args.in_data_folder):
        in_subfolder_path = os.path.join(args.in_data_folder, subfolder)
        if os.path.isdir(in_subfolder_path):
            augment_count = augment_counts.get(subfolder[:-4], 1)  # Default to 1 if not found
            print(f"Processing {subfolder} with {augment_count} augmentations...")
            process_folder(in_subfolder_path, args.out_data_folder, subfolder, args.mode, augment_count)

if __name__ == "__main__":
    main()
