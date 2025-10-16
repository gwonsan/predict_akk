import numpy as np
import pandas as pd
import h5py

def csv_to_mat(csv_path, width, height, output_path):
    # CSV 파일 읽기
    data = pd.read_csv(csv_path)
    
    # 데이터 형태 변환 (width*height, 152) -> (width, height, 151) 및 (width, height)
    data_3d = data.iloc[:, :150].values.reshape(height, width, 150)
    gt_2d = data.iloc[:, 150].values.reshape(height, width)
    
    # gt_2d 값 변환 (0->1, 1->2)
    gt_2d = gt_2d + 1
    
    # h5 파일로 저장
    with h5py.File(output_path + 'gongsan.h5', 'w') as f:
        f.create_dataset('input', data=data_3d, compression='gzip', compression_opts=9)
        f.create_dataset('gt', data=gt_2d, compression='gzip', compression_opts=9)
    
    print(f"저장 완료:\n데이터 크기: {data_3d.shape}\nGT 크기: {gt_2d.shape}")

if __name__ == "__main__":
    csv_to_mat('./gs_2307_sub_m1.csv',2563,2799,'./')