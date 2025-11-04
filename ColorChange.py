import numpy as np
import os
import random

class ColorChange:
    def __init__(self, in_data_folder, out_data_folder, exclude_class=None):
        self.in_data_folder = in_data_folder
        self.out_data_folder = out_data_folder
        self.exclude_class = exclude_class
        self.coord = None
        self.color = None
        self.segment = None
        self.load_data()

    def load_data(self):
        self.coord = np.load(os.path.join(self.in_data_folder, 'coord.npy'))
        self.color = np.load(os.path.join(self.in_data_folder, 'color.npy'))
        self.segment = np.load(os.path.join(self.in_data_folder, 'segment.npy'))

    def change_color(self, color):
        direction = random.choice(['RG', 'GB', 'BR'])  # random direction
        opp_color = color.copy()

        if direction == 'RG':
            opp_color[0] = 255 - color[1]  # R channel to G
            opp_color[1] = 255 - color[0]  # G channel to R
        elif direction == 'GB':
            opp_color[1] = 255 - color[2]  # G channel to B
            opp_color[2] = 255 - color[1]  # B channel to G
        elif direction == 'BR':
            opp_color[2] = 255 - color[0]  # B channel to R
            opp_color[0] = 255 - color[2]  # R channel to B

        # add random change
        new_color = np.clip(opp_color + np.random.randint(-70, 70, opp_color.shape), 0, 255)
        return new_color

    def change_color_class(self):
        unique, counts = np.unique(self.segment, return_counts=True)
        minority_classes = unique[np.argsort(counts)]

        augmented_coords = []
        augmented_colors = []
        augmented_segments = []

        for minority_class in minority_classes:
            if minority_class == self.exclude_class:
                print(f"Skipping color change for class {minority_class}.")
                continue

            print(f"Changing colors for class {minority_class}.")
            minority_data = np.hstack((self.coord, self.color, self.segment.reshape(-1, 1)))
            minority_data = minority_data[minority_data[:, -1] == minority_class]

            if len(minority_data) == 0:
                continue

            new_points = []
            color_delta = None

            for i in range(len(minority_data)):
                original_color = minority_data[i, 3:6]

                if color_delta is None:
                    # chage the first point
                    new_color = self.change_color(original_color)
                    color_delta = new_color - original_color  
                else:                    
                    new_color = np.clip(original_color + color_delta, 0, 255)

                new_point = np.concatenate((minority_data[i, :3], new_color, [minority_class]))
                new_points.append(new_point)

            new_points_array = np.array(new_points)
            augmented_coords.append(new_points_array[:, :3])
            augmented_colors.append(new_points_array[:, 3:6])
            augmented_segments.append(new_points_array[:, 6])

        # exclude_class append
        if self.exclude_class is not None:
            unmodified_data = np.hstack((self.coord, self.color, self.segment.reshape(-1, 1)))
            unmodified_data = unmodified_data[unmodified_data[:, -1] == self.exclude_class]

            augmented_coords.append(unmodified_data[:, :3])
            augmented_colors.append(unmodified_data[:, 3:6])
            augmented_segments.append(unmodified_data[:, 6])

        # save data
        self.coord = np.vstack(augmented_coords)
        self.color = np.vstack(augmented_colors)
        self.segment = np.hstack(augmented_segments)

        np.save(os.path.join(self.out_data_folder, 'augmented_coord.npy'), self.coord)
        np.save(os.path.join(self.out_data_folder, 'augmented_color.npy'), self.color)
        np.save(os.path.join(self.out_data_folder, 'augmented_segment.npy'), self.segment)

        print("Color change completed successfully.")
        return self.coord, self.color, self.segment
