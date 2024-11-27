# Place dataset seprately in train, val, txt folder


import os
import shutil

def copy_folders(src_base_path, dest_base_path, txt_file, folder_name):
    # txt 파일 읽기
    with open(txt_file, 'r') as f:
        folder_list = [line.strip() for line in f.readlines()]

    # 대상 폴더 경로가 없으면 생성
    dest_folder = os.path.join(dest_base_path, folder_name)
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # 각 폴더에 대해 파일 복사
    for folder in folder_list:
        src_folder = os.path.join(src_base_path, folder)
        if os.path.exists(src_folder):
            dest_subfolder = os.path.join(dest_folder, folder)
            if not os.path.exists(dest_subfolder):
                os.makedirs(dest_subfolder)

            # npy 파일들 복사
            for file in os.listdir(src_folder):
                if file.endswith('.npy'):
                    src_file = os.path.join(src_folder, file)
                    dest_file = os.path.join(dest_subfolder, file)
                    shutil.copy2(src_file, dest_file)
                    print(f"Copied {src_file} to {dest_file}")
        else:
            print(f"Folder {src_folder} does not exist")

if __name__ == "__main__":
    # 경로 설정
    source_base_path = '/data/volme/data/pointcept_data/volmeda_pointcept_all/volmeda_all'
    destination_base_path = '/data/volme/data/pointcept_data/volmeda_pointcept_all/volmeda_model/volmeda'

    # txt 파일 경로
    #train_txt = '/data/volme/3D_DataAugmentation/data_prepare/data_meta/aug/train.txt'
    #val_txt = '/data/volme/3D_DataAugmentation/data_prepare/data_meta/aug/val.txt'
    #test_txt = '/data/volme/3D_DataAugmentation/data_prepare/data_meta/aug/test.txt'
    train_txt = '/data/volme/3D_DataAugmentation/data_prepare/data_meta/original/train.txt'
    val_txt = '/data/volme/3D_DataAugmentation/data_prepare/data_meta/original/val.txt'
    test_txt = '/data/volme/3D_DataAugmentation/data_prepare/data_meta/original/test.txt'

    # 각 txt 파일에 맞춰 폴더 복사
    copy_folders(source_base_path, destination_base_path, train_txt, 'train')
    copy_folders(source_base_path, destination_base_path, val_txt, 'val')
    copy_folders(source_base_path, destination_base_path, test_txt, 'test')
