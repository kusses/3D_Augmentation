import numpy as np
import argparse

def analyze_npy_file(file_path):
    # npy 파일 로드
    data = np.load(file_path)

    # 데이터셋의 크기 표
    dataset_size = data.shape[0]
    print(f"데이터셋 크기: {dataset_size}")

    # 데이터셋의 차원 표
    dataset_dimension = data.ndim
    print(f"데이터셋 차원: {dataset_dimension}")

    # 데이터셋의 내용 살펴보기
    print("데이터셋 내용:")
    print(data)

if __name__ == "__main__":
    # argparse를 사용하여 파일 경로를 받기
    parser = argparse.ArgumentParser(description='Analyze npy file')
    parser.add_argument('file_path', type=str, help='Path to the npy file')
    args = parser.parse_args()

    # 분석 함수 호출
    analyze_npy_file(args.file_path)
