import os

def add_ply_to_folders(root_dir):
    # root_dir 내의 모든 폴더들을 확인
    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)

        # 폴더인지 확인
        if os.path.isdir(folder_path):
            # .ply로 끝나지 않는 폴더명에만 적용
            if not folder_name.endswith(".ply"):
                new_folder_name = folder_name + ".ply"
                new_folder_path = os.path.join(root_dir, new_folder_name)

                # 폴더 이름 변경
                os.rename(folder_path, new_folder_path)
                print(f"폴더 이름 변경: {folder_name} -> {new_folder_name}")
            else:
                print(f"이미 '.ply'가 붙은 폴더: {folder_name}")

if __name__ == "__main__":
    # 폴더들의 최상위 디렉토리 경로를 지정
    root_directory = "/data/volme/data/pointcept_data/volmeda_pointcept_all/volmeda/train"  # 여기에 폴더가 위치한 경로를 입력
    add_ply_to_folders(root_directory)
