"""
Script Overview:
This script provides utility functions for visualizing and converting 3D point cloud data stored in NumPy arrays to PLY files. 
It includes two main functions:
1. `visualize_ply(ply_file)`: Visualizes a PLY file using Open3D.
2. `numpy_to_ply(output_file, coord_file, color_file=None, segment_file=None)`: Converts NumPy arrays containing coordinates, 
   colors, and optional segmentation data into a PLY file format.

The script is designed to handle point cloud data with or without color and segmentation information, making it versatile for 
various 3D data processing tasks.

Code written by ETRI.
"""

import open3d as o3d
import plyfile
import numpy as np

def visualize_ply(ply_file):
    pcd = o3d.io.read_point_cloud(ply_file)
    o3d.visualization.draw_geometries([pcd])

def numpy_to_ply(output_file, coord_file, color_file=None, segment_file=None):
    # Load coordinates
    coords = np.load(coord_file)
    if coords is None or coords.shape[1] != 3:
        raise ValueError("Coordinates should be a valid numpy array of shape (N, 3)")

    # Prepare the vertex data structure
    vertex = [
        (coords[i, 0], coords[i, 1], coords[i, 2]) for i in range(coords.shape[0])
    ]

    # Load colors if available and add to vertex
    if color_file:
        colors = np.load(color_file)
        if colors is not None and colors.shape[1] == 3:
            vertex = [
                (coords[i, 0], coords[i, 1], coords[i, 2], colors[i, 0], colors[i, 1], colors[i, 2])
                for i in range(coords.shape[0])
            ]
        else:
            raise ValueError("Colors should be a valid numpy array of shape (N, 3)")

    # Load segments if available and add to vertex
    if segment_file:
        segments = np.load(segment_file)
        if segments is not None and segments.shape[0] == coords.shape[0]:
            vertex = [
                (coords[i, 0], coords[i, 1], coords[i, 2],
                 colors[i, 0], colors[i, 1], colors[i, 2], segments[i])
                for i in range(coords.shape[0])
            ]
        else:
            raise ValueError("Segments should be a valid numpy array of shape (N,)")

    # Define the PLY data structure
    vertex_dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
        ('class', 'i4')
    ] if segment_file else [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
    ]

    # Convert vertex list to numpy array
    vertex = np.array(vertex, dtype=vertex_dtype)

    # Create a PlyElement for vertex data
    ply_element = plyfile.PlyElement.describe(vertex, 'vertex')

    # Write to PLY file
    ply_data = plyfile.PlyData([ply_element], text=True)
    ply_data.write(output_file)
    print(f"Saved {output_file} successfully.")
