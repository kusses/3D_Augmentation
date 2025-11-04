import open3d as o3d
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
import time


def gt_visualize_numpy(gt_path, save_path):
    # Construct the required file paths
    coord_file, segment_file = None, None
    for fname in os.listdir(gt_path):
        if 'coord' in fname and fname.endswith('.npy'):
            coord_file = os.path.join(gt_path, fname)
            break
    for fname in os.listdir(gt_path):
        if 'segment' in fname and fname.endswith('.npy'):
            segment_file = os.path.join(gt_path, fname)
            break

    if coord_file is None or segment_file is None:
        print(f"Error: Required coordinate or segment file not found in '{gt_path}'.")
        return

    # Create save directory if it does not exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Load NumPy files
    coords = np.load(coord_file)
    segments = np.load(segment_file).astype(np.int32)

    # Check the number of points
    if coords.shape[0] != segments.shape[0]:
        print("Error: The number of coordinates does not match the number of segments.")
        return

    # Extract unique segment classes
    unique_segments = np.unique(segments)
    print(f"Number of unique segments: {len(unique_segments)}")

    # Define colors for each segment class
    num_classes = len(unique_segments)
    color_map = plt.get_cmap("tab10")
    colors = np.zeros((segments.shape[0], 3))

    for i, seg_class in enumerate(unique_segments):
        colors[segments == seg_class] = np.array(color_map(i / num_classes)[:3]) * 0.7

    # Create Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Save the point cloud visualization as a PNG file
    save_png(pcd, save_path, gt_path)


def save_png(pcd, save_path, file_path):
    # Extract the base name from file_path without extension
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    # Combine with save_path to create the save filename
    save_filename = f"{base_name}_{os.path.basename(save_path)}.png"
    save_filepath = os.path.join(save_path, save_filename)

    # Create visualizer and render to PNG
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True, width=600, height=600)
    vis.add_geometry(pcd)
    view_control = vis.get_view_control()
    view_control.set_zoom(0.8)  # Adjust zoom to fit entire point cloud
    view_control.rotate(180.0, 0.0)  # Rotate the view 180 degrees to show the back side
    vis.poll_events()
    time.sleep(1)  # Ensure proper rendering before capture
    vis.update_renderer()
    vis.capture_screen_image(save_filepath)
    vis.destroy_window()
    print(f"Saved visualization to {save_filepath}")


def pred_visualize_ply(pred_folder, gt_folder, save_path):
    # Iterate through all prediction files
    for pred_file in os.listdir(pred_folder):
        if pred_file.endswith('.ply_pred.npy'):
            base_name = pred_file.split('_pred.npy')[0]

            # Find the corresponding GT folder (train, val, test) and file
            for subdir in ['train', 'val', 'test']:
                gt_subfolder = os.path.join(gt_folder, subdir, base_name)
                if os.path.exists(gt_subfolder):
                    coord_folder = gt_subfolder  # Assigning the correct path for coord file search
                    pred_file_path = os.path.join(pred_folder, pred_file)
                    print(pred_file_path)
                    save_dir = os.path.join(save_path)

                    # Visualize GT and Pred
                    gt_visualize_numpy(coord_folder, os.path.join(save_dir, 'gt'))
                    pred_visualize_individual(coord_folder, pred_file_path, os.path.join(save_dir))
                    break


def pred_visualize_individual(file_path, pred_file_path, save_path):
    # Construct the required file paths
    coord_file = None
    for fname in os.listdir(file_path):
        if 'coord' in fname and fname.endswith('.npy'):
            coord_file = os.path.join(file_path, fname)
            break

    if coord_file is None:
        print("Error: No coordinate file found in the specified path.")
        return

    # Check if files exist
    if not os.path.exists(coord_file):
        print(f"Error: The coordinate file '{coord_file}' does not exist.")
        return
    if not os.path.exists(pred_file_path):
        print(f"Error: The prediction file '{pred_file_path}' does not exist.")
        return

    # Create save directory if it does not exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Load NumPy files
    coords = np.load(coord_file)
    pred_classes = np.load(pred_file_path).astype(np.int32)

    # Check the number of points
    if coords.shape[0] != pred_classes.shape[0]:
        print("Error: The number of coordinates does not match the number of predicted classes.")
        return

    # Extract unique predicted classes
    unique_classes = np.unique(pred_classes)
    print(f"Number of unique predicted classes: {len(unique_classes)}")

    # Define colors for each predicted class
    num_classes = len(unique_classes)
    color_map = plt.get_cmap("tab10")
    colors = np.zeros((pred_classes.shape[0], 3))

    for i, pred_class in enumerate(unique_classes):
        colors[pred_classes == pred_class] = color_map(i / num_classes)[:3]

    # Create Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Save the point cloud visualization as a PNG file
    save_png(pcd, save_path, file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize PLY files and class predictions")
    parser.add_argument("gt_folder", type=str, nargs='?',
                        default='/data/volme/data/pointcept_data/volmeda_pointcept_all/volmeda_model/volmeda',
                        help="Path to the GT folders (train, val, test)")
    parser.add_argument("pred_folder", type=str, nargs='?',
                        default='/data/volme/Pointcept/exp/volmeda/semseg-pt-v1m1/result',
                        help="Path to the prediction folder containing .ply_pred.npy files")
    parser.add_argument("save_path", type=str, nargs='?',
                        default='/data/volme/data/pointcept_data/volmeda_pointcept_all/result_capture/original',
                        help="Path to the capture save path")
    args = parser.parse_args()

    # Update to pass the correct save path
    pred_visualize_ply(args.pred_folder, args.gt_folder, args.save_path)
