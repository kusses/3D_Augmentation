"""
Script Description:
This script implements a 3D version of the SMOTE (Synthetic Minority Over-sampling Technique) algorithm.
The primary goal of this script is to augment minority classes within 3D point cloud datasets by generating new points based
on the nearest neighbors of existing points. Additionally, the script can apply color augmentation to the generated points,
providing more diversity in the dataset. The augmented data is then saved in NumPy format for further use.

Code written ETRI.
"""

import numpy as np
import os
import random
from sklearn.neighbors import NearestNeighbors

class SMOTE3D:
    def __init__(self, in_data_folder, out_data_folder, exclude_class=None, apply_color_augmentation=True):
        self.in_data_folder = in_data_folder
        self.out_data_folder = out_data_folder
        self.exclude_class = exclude_class
        self.apply_color_augmentation = apply_color_augmentation  # Flag to apply color augmentation
        self.coord = None
        self.color = None
        self.segment = None
        self.load_data()

    def load_data(self):
        """Load coordinate, color, and segment data from the input folder."""
        self.coord = np.load(os.path.join(self.in_data_folder, 'coord.npy'))
        self.color = np.load(os.path.join(self.in_data_folder, 'color.npy'))
        self.segment = np.load(os.path.join(self.in_data_folder, 'segment.npy'))

    def augment_coordinates(self, coords):
        """Apply a random shift to coordinates along the x, y, and z axes."""
        shift = np.array([random.uniform(-0.02, 0.02),  # x-axis shift
                          random.uniform(-0.02, 0.02),  # y-axis shift
                          random.uniform(-0.02, 0.02)])  # z-axis shift
        coords = coords + shift
        return coords

    def augment_colors(self, colors):
        """Apply color augmentation by transforming the RGB channels."""
        direction = random.choice(['RG', 'GB', 'BR'])  # Select one of three directions
        opp_colors = colors.copy()

        if direction == 'RG':
            opp_colors[0] = 255 - colors[1]  # Set R channel to the opposite of G
            opp_colors[1] = 255 - colors[0]  # Set G channel to the opposite of R
        elif direction == 'GB':
            opp_colors[1] = 255 - colors[2]  # Set G channel to the opposite of B
            opp_colors[2] = 255 - colors[1]  # Set B channel to the opposite of G
        elif direction == 'BR':
            opp_colors[2] = 255 - colors[0]  # Set B channel to the opposite of R
            opp_colors[0] = 255 - colors[2]  # Set R channel to the opposite of B

        # Add random variation to the transformed colors
        new_colors = np.clip(opp_colors + np.random.randint(-70, 70, opp_colors.shape), 0, 255)
        return new_colors

    def augment_minority_class(self, k_neighbors=5, augmentation_factor=2):
        """Augment the minority classes by generating new points based on the nearest neighbors."""
        unique, counts = np.unique(self.segment, return_counts=True)
        minority_classes = unique[np.argsort(counts)]

        for minority_class in minority_classes:
            if minority_class == self.exclude_class:
                print(f"Skipping augmentation for class {minority_class}.")
                continue

            print(f"Augmenting class {minority_class}.")
            minority_data = np.hstack((self.coord, self.color, self.segment.reshape(-1, 1)))
            minority_data = minority_data[minority_data[:, -1] == minority_class]

            nbrs = NearestNeighbors(n_neighbors=k_neighbors).fit(minority_data[:, :3])
            _, indices = nbrs.kneighbors(minority_data[:, :3])

            new_points = []
            color_delta = None  # Variable to store color change

            for i in range(len(minority_data)):
                for idx in indices[i]:
                    if idx != i:
                        for _ in range(augmentation_factor):  # Increase augmentation factor
                            # Coordinate augmentation
                            new_coord = self.augment_coordinates(minority_data[i, :3])
                            # Color augmentation based on the option
                            if self.apply_color_augmentation:
                                # Check current color
                                original_color = minority_data[i, 3:6]

                                if color_delta is None:
                                    # Perform color transformation on the first point
                                    new_color = self.augment_colors(original_color)
                                    color_delta = new_color - original_color  # Save the color change
                                else:
                                    # Apply the same change to other points
                                    new_color = original_color + color_delta
                                    new_color = np.clip(new_color, 0, 255)  # Limit color values to the 0-255 range

                            else:
                                new_color = minority_data[i, 3:6]

                            new_point = np.concatenate((new_coord, new_color, [minority_class]))
                            new_points.append(new_point)

            new_points_array = np.array(new_points)
            augmented_data = np.vstack(
                (np.hstack((self.coord, self.color, self.segment.reshape(-1, 1))), new_points_array))

            self.coord = augmented_data[:, :3]
            self.color = augmented_data[:, 3:6]
            self.segment = augmented_data[:, 6]

        # Save the augmented data to the output folder
        np.save(os.path.join(self.out_data_folder, 'augmented_coord.npy'), self.coord)
        np.save(os.path.join(self.out_data_folder, 'augmented_color.npy'), self.color)
        np.save(os.path.join(self.out_data_folder, 'augmented_segment.npy'), self.segment)

        print("Augmentation completed successfully.")
        return self.coord, self.color, self.segment
