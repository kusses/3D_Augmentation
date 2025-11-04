import os
import numpy as np
from scipy.spatial import KDTree


def vertex_normal(coords, k=10):
    """주어진 좌표에서 노멀 벡터를 계산"""
    kdtree = KDTree(coords)
    normals = np.zeros_like(coords)

    for i, coord in enumerate(coords):
        distances, indices = kdtree.query(coord, k=k)
        neighbors = coords[indices[1:]]  # 첫 번째 이웃은 자기 자신이므로 제외
        cov_matrix = np.cov(neighbors - coord, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov_matrix)
        normals[i] = eigvecs[:, 0]  # 최소 고유값에 대응하는 고유벡터가 노멀 벡터

    return normals


def remove_augmented_normal_file(folder_path):
    """폴더 내에 augmented_coord_normal.npy 파일이 있을 경우 이를 삭제"""
    augmented_normal_file = os.path.join(folder_path, "augmented_coord_normal.npy")
    if os.path.exists(augmented_normal_file):
        try:
            os.remove(augmented_normal_file)
            print(f"Deleted file: {augmented_normal_file}")
        except Exception as e:
            print(f"Error deleting {augmented_normal_file}: {e}")
    else:
        print(f"No augmented_coord_normal.npy file found in {folder_path}")

def generate_normal_file(folder_path):
    """폴더 내 *normal*.npy가 없으면 *coord*.npy 또는 *augmented_coord*.npy를 사용하여 노멀 파일 생성"""
    normal_files = [f for f in os.listdir(folder_path) if 'normal' in f and f.endswith('.npy')]
    coord_files = [f for f in os.listdir(folder_path) if 'coord' in f and f.endswith('.npy')]  # augmented_coord도 포함

    # *normal*.npy 파일이 없을 경우에만 처리
    if not normal_files and coord_files:
        for coord_file in coord_files:
            try:
                coord_path = os.path.join(folder_path, coord_file)
                coords = np.load(coord_path)

                print(f"Loaded coords from {coord_file}: {coords.shape}")  # 디버깅 출력

                # 좌표 데이터가 올바른 형식(2D, 3열)이면 노멀 벡터 생성
                if isinstance(coords, np.ndarray) and coords.ndim == 2 and coords.shape[1] == 3:
                    normals = vertex_normal(coords)
                    normal_path = os.path.join(folder_path, "normal.npy")  # 파일 이름을 normal.npy로 고정
                    np.save(normal_path, normals)
                    print(f"Generated and saved normals to {normal_path}")
                else:
                    print(f"Invalid coord format in {coord_file}. Expected a 2D array with 3 columns.")
            except Exception as e:
                print(f"Error processing {coord_file}: {e}")
    else:
        if normal_files:
            print(f"Normal file already exists in {folder_path}")
        else:
            print(f"No coord files found in {folder_path}")


def process_folders(root_dir):
    """각 폴더를 순회하며 *normal*.npy 파일 생성 및 augmented_coord_normal.npy 파일 삭제"""
    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)

        # 폴더일 경우에만 처리
        if os.path.isdir(folder_path):
            print(f"Processing folder: {folder_name}")
            # 먼저 augmented_coord_normal.npy 파일이 있으면 삭제
            remove_augmented_normal_file(folder_path)
            # 그 후 normal.npy 파일 생성
            generate_normal_file(folder_path)


if __name__ == "__main__":
    root_directory = "/data/volme/data/pointcept_data/volmeda_pointcept_all/volmeda_all"  # 최상위 폴더 경로
    process_folders(root_directory)
