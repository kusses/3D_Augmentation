# mix3d_npy_augmentation.py
import numpy as np
import os
from pathlib import Path
import random
from tqdm import tqdm
import shutil


class Mix3DNPYAugmentor:
    def __init__(self, data_root, mix_ratio=5, seed=42):
        """
        data_root: coord.npy, color.npy, normal.npy, segment.npy scene folders
        mix_ratio: number of other scenes mixing with each scene
        """
        self.data_root = Path(data_root)
        self.output_root = self.data_root / 'mix3d_augmented'
        self.output_root.mkdir(exist_ok=True, parents=True)
        self.mix_ratio = mix_ratio
        random.seed(seed)
        np.random.seed(seed)

    def load_scene(self, scene_path):
        """Scene data load"""
        data = {}
        data['coord'] = np.load(scene_path / 'coord.npy')  # (N, 3)
        data['color'] = np.load(scene_path / 'color.npy')  # (N, 3)
        data['normal'] = np.load(scene_path / 'normal.npy')  # (N, 3)
        data['segment'] = np.load(scene_path / 'segment.npy')  # (N,)
        return data

    def augment_scene(self, data):
        """Mix3D augmentation"""
        coord = data['coord'].copy()
        color = data['color'].copy()
        normal = data['normal'].copy()

        # 1. Center to origin
        center = coord.mean(axis=0)
        coord = coord - center

        # 2. Random rotation (z-axis main)
        angle_z = np.random.uniform(0, 2 * np.pi)
        cos_z, sin_z = np.cos(angle_z), np.sin(angle_z)
        rot_z = np.array([[cos_z, -sin_z, 0],
                          [sin_z, cos_z, 0],
                          [0, 0, 1]])

        # 3. Small random rotation (x,y axes)
        angle_x = np.random.uniform(-np.pi / 64, np.pi / 64)
        angle_y = np.random.uniform(-np.pi / 64, np.pi / 64)

        cos_x, sin_x = np.cos(angle_x), np.sin(angle_x)
        rot_x = np.array([[1, 0, 0],
                          [0, cos_x, -sin_x],
                          [0, sin_x, cos_x]])

        cos_y, sin_y = np.cos(angle_y), np.sin(angle_y)
        rot_y = np.array([[cos_y, 0, sin_y],
                          [0, 1, 0],
                          [-sin_y, 0, cos_y]])

        # rotation matrix
        rotation = rot_z @ rot_x @ rot_y

        # applying rotation
        coord = coord @ rotation.T
        normal = normal @ rotation.T

        # 4. Random scale
        scale = np.random.uniform(0.9, 1.1)
        coord = coord * scale

        # 5. Random flip
        if np.random.rand() > 0.5:
            coord[:, 0] *= -1
            normal[:, 0] *= -1
        if np.random.rand() > 0.5:
            coord[:, 1] *= -1
            normal[:, 1] *= -1

        # 6. Color augmentation
        # Brightness
        brightness = np.random.uniform(-0.1, 0.1)
        color = color + brightness

        # Contrast
        contrast = np.random.uniform(0.8, 1.2)
        color = (color - 0.5) * contrast + 0.5

        # Saturation
        gray = color.mean(axis=1, keepdims=True)
        saturation = np.random.uniform(0.8, 1.2)
        color = gray + saturation * (color - gray)

        # Clipping
        color = np.clip(color, 0, 1)

        # 7. Random translation (small)
        translation = np.random.uniform(-0.1, 0.1, 3)
        coord = coord + translation

        return {
            'coord': coord,
            'color': color,
            'normal': normal,
            'segment': data['segment'].copy()
        }

    def mix_two_scenes(self, scene1_path, scene2_path):
        """mixing two scene through Mix3D"""
        # Load scenes
        data1 = self.load_scene(scene1_path)
        data2 = self.load_scene(scene2_path)

        # Augment each scene
        aug_data1 = self.augment_scene(data1)
        aug_data2 = self.augment_scene(data2)

        # Mix3D: concatenation
        mixed_data = {
            'coord': np.vstack([aug_data1['coord'], aug_data2['coord']]),
            'color': np.vstack([aug_data1['color'], aug_data2['color']]),
            'normal': np.vstack([aug_data1['normal'], aug_data2['normal']]),
            'segment': np.concatenate([aug_data1['segment'], aug_data2['segment']])
        }

        return mixed_data

    def save_scene(self, data, output_path):
        """Scene data save"""
        output_path.mkdir(exist_ok=True, parents=True)
        np.save(output_path / 'coord.npy', data['coord'])
        np.save(output_path / 'color.npy', data['color'])
        np.save(output_path / 'normal.npy', data['normal'])
        np.save(output_path / 'segment.npy', data['segment'])

    def process_dataset(self):
        """Dataset processing"""
        # Scene folder exploring (예: 3401.ply, 3402.ply ...)
        scene_dirs = sorted([d for d in self.data_root.iterdir()
                             if d.is_dir() and d.name.endswith('.ply')
                             and d.name != 'mix3d_augmented'])

        print(f"Found {len(scene_dirs)} scenes")
        print(f"Will create {len(scene_dirs) * self.mix_ratio} augmented scenes")

        # statistics
        total_mixed = 0

        # processing each scene
        for i, scene1_dir in enumerate(tqdm(scene_dirs, desc="Processing scenes")):
            # 다른 scene들 중에서 랜덤 선택
            other_scenes = [s for j, s in enumerate(scene_dirs) if j != i]

            # mix_ratio based random selection
            if len(other_scenes) >= self.mix_ratio:
                selected_scenes = random.sample(other_scenes, self.mix_ratio)
            else:
                selected_scenes = other_scenes

            for j, scene2_dir in enumerate(selected_scenes):
                # Mix two scenes
                mixed_data = self.mix_two_scenes(scene1_dir, scene2_dir)

                # 출력 경로 생성 (원본과 동일한 폴더 구조 유지)
                # 예: 3401_mixed_3402.ply
                scene1_name = scene1_dir.name.replace('.ply', '')
                scene2_name = scene2_dir.name.replace('.ply', '')
                output_name = f"{scene1_name}_mixed_{scene2_name}.ply"
                output_path = self.output_root / output_name

                # 저장
                self.save_scene(mixed_data, output_path)
                total_mixed += 1

        print(f"\n✅ Created {total_mixed} mixed scenes in {self.output_root}")


    def validate_augmentation(self):
        print("\n Validating augmented data...")

        augmented_scenes = list(self.output_root.iterdir())
        print(f"Total augmented scenes: {len(augmented_scenes)}")

        # 샘플 scene 검사
        if augmented_scenes:
            sample_scene = augmented_scenes[0]
            data = self.load_scene(sample_scene)

            print(f"\nSample scene: {sample_scene.name}")
            print(f"  Points: {len(data['coord'])}")
            print(f"  Coord shape: {data['coord'].shape}")
            print(f"  Color shape: {data['color'].shape}")
            print(f"  Normal shape: {data['normal'].shape}")
            print(f"  Segment shape: {data['segment'].shape}")
            print(f"  Unique labels: {len(np.unique(data['segment']))}")


# 실행 코드
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Mix3D Data Augmentation for NPY format')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory containing scene folders')
    parser.add_argument('--mix_ratio', type=int, default=5,
                        help='Number of scenes to mix with each scene')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    # Mix
    augmentor = Mix3DNPYAugmentor(
        data_root=args.data_root,
        mix_ratio=args.mix_ratio,
        seed=args.seed
    )

    augmentor.process_dataset()

    augmentor.validate_augmentation()