import argparse
import os
import rasterio
import geopandas as gpd
import numpy as np
from rasterio.features import geometry_mask
import h5py
from skimage.transform import resize ### 추가된 라이브러리
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

### downsample_factor 인자를 함수에 추가
def create_indian_pine_format_data(akk_shp_file_path, akk_shp_file_name, gt_shp_file_path, gt_shp_file_name, hsp_img_path, hsp_img_name, output_h5_path, output_h5_file, downsample_factor=1):
    # 1. Shapefile (라벨 데이터) 읽기
    akk_shp_data = gpd.read_file(os.path.join(akk_shp_file_path, akk_shp_file_name))
    gt_shp_data = gpd.read_file(os.path.join(gt_shp_file_path, gt_shp_file_name))
    
    logger.info(f"Shapefile data: {akk_shp_data.shape}")

    # 2. 초분광 이미지 데이터 읽기
    hsp_img_file = os.path.join(hsp_img_path, hsp_img_name)
    with rasterio.open(hsp_img_file) as src:
        hsp_data = src.read()  # Shape: (bands, height, width)
        transform = src.transform
        hsp_meta = src.meta

    logger.info(f"Original Hyperspectral image data shape: {hsp_data.shape}")

    # 3. 초분광 이미지의 차원 조정 (C, H, W -> H, W, C 형태로 변환)
    hsp_data = np.transpose(hsp_data, (1, 2, 0))  # (H, W, C)
    
    # 4. TR 및 TE 초기화 (기본값: 0)
    TR = np.zeros((hsp_data.shape[0], hsp_data.shape[1]), dtype=np.uint8)
    TE = np.zeros((hsp_data.shape[0], hsp_data.shape[1]), dtype=np.uint8)

    # 5. 각 폴리곤에 대해 TR과 TE 구성 (기존 코드와 동일)
    np.random.seed(42)
    gt_train_sample = 0
    for idx in range(len(akk_shp_data)):
        geom = [akk_shp_data.iloc[idx]['geometry']]
        mask = geometry_mask(geom, transform=transform, invert=True, out_shape=(hsp_data.shape[0], hsp_data.shape[1]))
        polygon_indices = np.argwhere(mask)
        num_total_samples = len(polygon_indices)
        num_train_samples = int(0.4 * num_total_samples)
        gt_train_sample += num_train_samples
        if num_total_samples > 0:
            train_indices_indices = np.random.choice(num_total_samples, num_train_samples, replace=False)
            train_indices = polygon_indices[train_indices_indices]
            for train_idx in train_indices:
                TR[train_idx[0], train_idx[1]] = 1
            
            test_indices_mask = np.ones(num_total_samples, dtype=bool)
            test_indices_mask[train_indices_indices] = False
            test_indices = polygon_indices[test_indices_mask]
            for test_idx in test_indices:
                if TR[test_idx[0], test_idx[1]] == 0:
                    TE[test_idx[0], test_idx[1]] = 1

    for idx in range(len(gt_shp_data)):
        gt_geom = [gt_shp_data.iloc[idx]['geometry']]
        gt_mask = geometry_mask(gt_geom, transform=transform, invert=True, out_shape=(hsp_data.shape[0], hsp_data.shape[1]))
        polygon_indices = np.argwhere(gt_mask)
        num_total_samples = len(polygon_indices)
        if args.method == 'same':
            num_train_samples = int(gt_train_sample / 2)
        else:
            num_train_samples = int(0.2 * num_total_samples)
        
        if num_total_samples > 0:
            train_indices_indices = np.random.choice(num_total_samples, num_train_samples, replace=False)
            train_indices = polygon_indices[train_indices_indices]
            for train_idx in train_indices:
                TR[train_idx[0], train_idx[1]] = 2
            
            test_indices_mask = np.ones(num_total_samples, dtype=bool)
            test_indices_mask[train_indices_indices] = False
            test_indices = polygon_indices[test_indices_mask]
            for test_idx in test_indices:
                if TR[test_idx[0], test_idx[1]] == 0:
                    TE[test_idx[0], test_idx[1]] = 2
    
    ### ---------------------------------------------------- ###
    ### 6. 다운샘플링 처리 (추가된 부분)                      ###
    ### ---------------------------------------------------- ###
    if downsample_factor > 1:
        logger.info(f"--- Applying downsampling with factor: {downsample_factor} ---")
        original_height, original_width, _ = hsp_data.shape
        new_height = original_height // downsample_factor
        new_width = original_width // downsample_factor

        # (A) 초분광 이미지 다운샘플링 (Bilinear Interpolation)
        logger.info(f"Downsampling hyperspectral image to ({new_height}, {new_width})")
        hsp_data = resize(
            hsp_data,
            (new_height, new_width),
            order=1,  # 1: Bilinear, 3: Bicubic
            preserve_range=True,
            anti_aliasing=True
        ).astype(hsp_data.dtype)

        # (B) TR, TE 마스크 다운샘플링 (Nearest Neighbor)
        logger.info(f"Downsampling TR/TE masks to ({new_height}, {new_width})")
        TR = resize(
            TR,
            (new_height, new_width),
            order=0,  # 0: Nearest neighbor (for labels)
            preserve_range=True,
            anti_aliasing=False
        ).astype(TR.dtype)
        
        TE = resize(
            TE,
            (new_height, new_width),
            order=0,
            preserve_range=True,
            anti_aliasing=False
        ).astype(TE.dtype)

        # 다운샘플링된 파일명 생성
        name, ext = os.path.splitext(output_h5_file)
        output_h5_file = f"{name}_x{downsample_factor}{ext}"
        
    logger.info(f'Final data shape: {hsp_data.shape}')
    logger.info(f'Final TR shape: {TR.shape}')
    logger.info(f'Final TE shape: {TE.shape}')

    # 7. 데이터 정규화
    input_normalize = np.zeros(hsp_data.shape)
    for i in range(hsp_data.shape[2]):
        input_max = np.max(hsp_data[:, :, i])
        input_min = np.min(hsp_data[:, :, i])
        if (input_max - input_min) != 0:
            input_normalize[:, :, i] = (hsp_data[:, :, i] - input_min) / (input_max - input_min)

    # 8. HDF5 형식으로 데이터 저장
    with h5py.File(os.path.join(output_h5_path, output_h5_file), 'w') as f:
        f.create_dataset('input', data=input_normalize, compression='gzip', compression_opts=9)
        f.create_dataset('TR', data=TR, compression='gzip', compression_opts=9)
        f.create_dataset('TE', data=TE, compression='gzip', compression_opts=9)

    logger.info(f"Data saved as '{os.path.join(output_h5_path, output_h5_file)}'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create Indian Pine format data from shapefile and hyperspectral image')
    # --- 기존 인자들은 그대로 유지 ---
    parser.add_argument('--akk_shp_file_path', type=str, default='../shp/', help='Path to the directory containing the shapefile')
    parser.add_argument('--akk_shp_file_name', type=str, default='GS_Akk_Polygon_240404__Merge.shp', help='Name of the shapefile')
    parser.add_argument('--gt_shp_file_path', type=str, default='../shp/gt/', help='Path to the directory containing the shapefile')
    parser.add_argument('--gt_shp_file_name', type=str, default='GS_23_6_gt_Polygon.shp', help='Name of the shapefile')
    parser.add_argument('--hsp_img_path', type=str, default='../img/gongsan', help='Path to the hyperspectral image')
    parser.add_argument('--hsp_img_name', type=str, default='20230703_Gongsan_Hyper', help='Name of the hyperspectral image')
    parser.add_argument('--output_h5_path', type=str, default='../h5_data', help='Path to save the output HDF5 file')
    parser.add_argument('--output_h5_file', type=str, default='20230703_GS.h5', help='Name of the output HDF5 file')
    parser.add_argument('--method', type=str, default='same')
    
    ### ---------------------------------------------------- ###
    ### 다운샘플링 인자 추가                                 ###
    ### ---------------------------------------------------- ###
    parser.add_argument('--downsample_factor',
                        type=int,
                        default=1,
                        help='Factor by which to downsample the spatial resolution (e.g., 2, 4, 8). Default is 1 (no downsampling).')
    
    args = parser.parse_args()
    
    create_indian_pine_format_data(
        args.akk_shp_file_path, 
        args.akk_shp_file_name, 
        args.gt_shp_file_path, 
        args.gt_shp_file_name, 
        args.hsp_img_path, 
        args.hsp_img_name, 
        args.output_h5_path, 
        args.output_h5_file,
        args.downsample_factor ### 추가된 인자 전달
    )