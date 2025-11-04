import os
import numpy as np
import json

# Function to create a PCD file from coordinates and colors
def write_pcd(file_path, coords, colors):
    num_points = coords.shape[0]

    # Header for PCD file
    header = f"""# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z rgb
SIZE 4 4 4 4
TYPE F F F F
COUNT 1 1 1 1
WIDTH {num_points}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {num_points}
DATA ascii
"""
    # Convert colors to packed RGB format
    colors_rgb = (colors[:, 0] << 16) | (colors[:, 1] << 8) | colors[:, 2]

    # Writing points and colors
    with open(file_path, 'w') as file:
        file.write(header)
        for i in range(num_points):
            x, y, z = coords[i]
            rgb = colors_rgb[i]
            file.write(f"{x} {y} {z} {rgb}\n")


# Function to rotate coordinates
def rotate_coords(coords, axis='x', angle_deg=90):
    angle_rad = np.deg2rad(angle_deg)
    if axis == 'x':
        rotation_matrix = np.array([[1, 0, 0],
                                    [0, np.cos(angle_rad), -np.sin(angle_rad)],
                                    [0, np.sin(angle_rad), np.cos(angle_rad)]])
    elif axis == 'y':
        rotation_matrix = np.array([[np.cos(angle_rad), 0, np.sin(angle_rad)],
                                    [0, 1, 0],
                                    [-np.sin(angle_rad), 0, np.cos(angle_rad)]])
    elif axis == 'z':
        rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad), 0],
                                    [np.sin(angle_rad), np.cos(angle_rad), 0],
                                    [0, 0, 1]])
    else:
        raise ValueError("Invalid axis. Choose from 'x', 'y', or 'z'.")

    # Apply rotation to coordinates
    return np.dot(coords, rotation_matrix.T)


# Main function to process all folders
def process_folders(base_directory, output_directory, segment_directory=None, pred=None):
    for folder_name in os.listdir(base_directory):
        folder_path = os.path.join(base_directory, folder_name)

        if not os.path.isdir(output_directory):
            os.mkdir(output_directory)
        # Skip if not a directory
        if not os.path.isdir(folder_path):
            continue

        # Construct file paths
        coord_path = os.path.join(folder_path, 'coord.npy')
        color_path = os.path.join(folder_path, 'color.npy')

        # Check if necessary files exist
        if not (os.path.exists(coord_path) and os.path.exists(color_path)):
            print(f"Skipping folder {folder_name}: Required files not found.")
            continue

        # Determine segment path
        if segment_directory:
            segment_file_name = f"{folder_name}_pred.npy"
            segment_path = os.path.join(segment_directory, segment_file_name)
        else:
            segment_path = os.path.join(folder_path, 'segment.npy')

        # Check if the segment file exists
        if not os.path.exists(segment_path):
            print(f"Skipping folder {folder_name}: Segment file not found.")
            continue

        # Load data
        coords = np.load(coord_path)
        colors = np.load(color_path)
        segments = np.load(segment_path)

        # Apply rotation to coordinates (e.g., 90 degrees around X-axis)
        coords = rotate_coords(coords, axis='x', angle_deg=90)

        # Create output directories for pointcloud and label
        pointcloud_dir = os.path.join(output_directory, 'pointcloud')
        label_dir = os.path.join(output_directory, 'label')
        os.makedirs(pointcloud_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)

        # Generate output file names
        output_base_name = os.path.splitext(folder_name)[0]
        pcd_file_path = os.path.join(pointcloud_dir, f"{output_base_name}.pcd")
        json_file_path = os.path.join(label_dir, f"{output_base_name}.json")

        # Create .pcd file
        write_pcd(pcd_file_path, coords, colors)

        # Convert segments to a list, and add 10000 if `pred` is provided
        segments_list = segments.flatten().tolist()
        if pred:
            segments_list = [seg + 10000 for seg in segments_list]

        # Save segments as JSON
        with open(json_file_path, 'w') as json_file:
            json.dump(segments_list, json_file)

        print(f"Processed folder {folder_name}: Created {pcd_file_path} and {json_file_path}")


# Example usage
base_directory = '/data/volme/data/pointcept_data/volmeda_pointcept/val'
segment_directory = '/data/volme/Pointcept/exp/volmeda/semseg-pt-v3m1-0-base/result_back'
output_directory = '/data/volme/data/pointcept_data/volmeda_pointcept/PRED'

# Call process_folders with pred parameter
process_folders(base_directory, output_directory, segment_directory=segment_directory, pred=True)
