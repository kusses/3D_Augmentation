import os
import numpy as np
import plyfile

# Define the paths
ply_folder_path = '/data/volme/data/VOLMEDA/volmeda'
npy_folder_path = '/data/volme/Pointcept/exp/volmeda/semseg-pt-v1m1/result'
output_folder_path = '/data/volme/data/VOLMEDA/prediction_result'

# Ensure the output directory exists
os.makedirs(output_folder_path, exist_ok=True)

# Get list of all PLY files in the folder
ply_files = [f for f in os.listdir(ply_folder_path) if f.endswith('.ply')]

# Process each PLY file
for ply_file in ply_files:
    ply_file_path = os.path.join(ply_folder_path, ply_file)
    base_name = os.path.splitext(ply_file)[0]

    # Find the corresponding NPY file
    npy_file_name = f"{base_name}.ply_pred.npy"
    npy_file_path = os.path.join(npy_folder_path, npy_file_name)

    if not os.path.exists(npy_file_path):
        print(f"NPY file not found for {ply_file}, skipping.")
        continue

    # Load the NPY file containing pred_class values
    pred_class_values = np.load(npy_file_path)

    # Read the PLY file
    ply_data = plyfile.PlyData.read(ply_file_path)

    # Ensure the number of pred_class values matches the number of vertices
    vertex_data = ply_data['vertex'].data
    if len(pred_class_values) != len(vertex_data):
        print(f"Mismatch in number of pred_class values for {ply_file}, skipping.")
        continue

    # Add the pred_class property to the vertex data
    new_vertex_data = np.array(
        [tuple(list(vertex) + [pred_class]) for vertex, pred_class in zip(vertex_data, pred_class_values)],
        dtype=vertex_data.dtype.descr + [('pred_class', 'i4')]
    )

    # Create a new PlyElement with the added property
    new_vertex_element = plyfile.PlyElement.describe(new_vertex_data, 'vertex')

    # Convert elements to list if it's a tuple
    elements_list = list(ply_data.elements)
    elements_list[0] = new_vertex_element  # replace the vertex element with the new one

    # Create a new PlyData object
    new_ply_data = plyfile.PlyData(elements_list, text=True)

    # Save the modified PLY file
    new_ply_file_path = os.path.join(output_folder_path, f"{base_name}_pred_all.ply")
    new_ply_data.write(new_ply_file_path)

    print(f"Processed and saved {new_ply_file_path}")
