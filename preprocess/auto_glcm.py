import numpy as np
import rasterio 
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import pandas as pd
import time

# ===============================
# 1. 데이터 로드 및 설정
# ===============================

start_time = time.time()  # 전체 실행 시간 측정 시작

# 초분광 영상의 크기: (2799, 2563, 150)
#부소산성
# hsi_height = 2715
# hsi_width = 1843

#공산성
hsi_height = 2799
hsi_width = 2563

num_bands = 150

# 초분광 데이터 로딩 (실제 코드에서는 파일 입출력을 사용)
with rasterio.open("/home/labhosik4609/gs/busosan/img/gongsan/03_20230703_Gongsan_Hyper_Subset") as src:
    data1 = src.read()
    hsi_image = data1.transpose((1,2,0))

print(f"초분광 영상 로딩 시간: {time.time() - start_time:.2f}초")

# RGB 영상 로드 (초분광 영상보다 10배 해상도)
scale = 10
tif_path = "/home/labhosik4609/gs/busosan/rgb_image/20230703_GS_sub_RGB_Mosaic.tif"
rgb_start_time = time.time()
with rasterio.open(tif_path) as src:
    raster = src.read()
    raster = raster.transpose((1,2,0))
    rgb_image = raster[:, :, :3]

print(f"RGB 영상 로딩 시간: {time.time() - rgb_start_time:.2f}초")

# ===============================
# 2. RGB 이미지 전처리
# ===============================
preprocess_start_time = time.time()

# RGB 이미지를 회색조로 변환
gray_image = rgb2gray(rgb_image)  # 결과 shape: (rgb_height, rgb_width)

# skimage의 graycomatrix는 정수 이미지를 필요로 하므로 0~255 범위의 uint8로 변환
gray_int = (gray_image * 255).astype(np.uint8)

print(f"전처리 소요 시간: {time.time() - preprocess_start_time:.2f}초")

# ===============================
# 3. 각 초분광 픽셀의 3x3 영역(자신 및 이웃)에 대한 RGB 블록에서 GLCM 특성 계산
# ===============================

glcm_start_time = time.time()

# 사용할 GLCM 특성 개수: 각 특성을 4방향에 대해 계산 → 총 4×4 = 16개
num_glcm_features = 16

# GLCM 계산 파라미터
distances = [1]
# 대각선 방향 (오른쪽 위, 오른쪽 아래, 왼쪽 아래, 왼쪽 위)
angles = [np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4]
levels = 256   # gray_int 값의 수준

def process_row(args):
    i, gray_int, scale, hsi_width = args
    row_features = np.zeros((hsi_width, num_glcm_features), dtype=np.float32)
    
    for j in range(hsi_width):
        # 초분광 픽셀 (i, j)를 기준으로 3×3 이웃에 해당하는 영역을 RGB 이미지에서 추출
        # hyperspectral grid에서 (i-1)부터 (i+1)까지, 즉 3픽셀에 해당하므로
        # 실제 RGB 이미지에서는 (i-1)*scale 부터 (i+2)*scale 까지 선택
        row_start = max(0, (i - 1) * scale)
        row_end   = min(gray_int.shape[0], (i + 2) * scale)  # i+2가 upper bound
        col_start = max(0, (j - 1) * scale)
        col_end   = min(gray_int.shape[1], (j + 2) * scale)
        
        block = gray_int[row_start:row_end, col_start:col_end]
        
        # 블록 크기가 너무 작으면(GT 최소 2×2) 건너뜁니다.
        if block.shape[0] < 2 or block.shape[1] < 2:
            continue
        
        # GLCM 계산: 결과 shape은 (levels, levels, len(distances), len(angles))
        glcm = graycomatrix(block, distances=distances, angles=angles,
                             levels=levels, symmetric=True, normed=True)
        
        # 각 방향에 대해 특성 계산 (거리 1개이므로 첫 번째 인덱스만 사용)
        contrast_vals    = graycoprops(glcm, 'contrast')[0, :]     # shape: (4,)
        energy_vals      = graycoprops(glcm, 'energy')[0, :]           # shape: (4,)
        homogeneity_vals = graycoprops(glcm, 'homogeneity')[0, :]      # shape: (4,)
        correlation_vals = graycoprops(glcm, 'correlation')[0, :]      # shape: (4,)
        
        # 네 가지 특성을 순서대로 연결하여 16차원 벡터 생성  
        features = np.concatenate([contrast_vals, energy_vals, homogeneity_vals, correlation_vals])
        row_features[j] = features
    
    return i, row_features

print("GLCM 특성 계산 시작...")

# 멀티프로세싱 설정
num_processes = cpu_count()  # CPU 코어 수만큼 프로세스 생성
pool = Pool(processes=num_processes)

# 초분광 영상의 각 행에 대해 작업 분배
args_list = [(i, gray_int, scale, hsi_width) for i in range(hsi_height)]
results = []

completed = 0
total = hsi_height

for result in pool.imap_unordered(process_row, args_list):
    completed += 1
    if completed % 100 == 0:
        print(f"진행률: {completed}/{total} ({(completed/total)*100:.2f}%)")
    results.append(result)

pool.close()
pool.join()

# 결과 정렬 및 배열 생성
glcm_features = np.zeros((hsi_height, hsi_width, num_glcm_features), dtype=np.float32)
for i, row_features in sorted(results):
    glcm_features[i] = row_features

print(f"GLCM 특성 계산 완료. 소요 시간: {time.time() - glcm_start_time:.2f}초")

# ===============================
# 4. 초분광 스펙트럼 특성과 GLCM 특성 결합
# ===============================
combine_start_time = time.time()

# hsi_image: (2799, 2563, 150)
# glcm_features: (2799, 2563, 16)
# 두 배열을 채널(axis=-1) 방향으로 결합
combined_image = np.concatenate((hsi_image, glcm_features), axis=-1)
# 최종 shape: (2799, 2563, 150+16)

# 2차원 피처 행렬로 변환 (각 행은 하나의 초분광 픽셀에 대한 특성)
combined_features = combined_image.reshape(-1, num_bands + num_glcm_features)
print("최종 피처 행렬 shape:", combined_features.shape)
print(f"특성 결합 소요 시간: {time.time() - combine_start_time:.2f}초")
# 예상 shape: (2799*2563, 166)

# ===============================
# 5. CSV 저장
# ===============================
save_start_time = time.time()

# 초분광 밴드 컬럼 이름
band_columns = [f'Band_{i+1}' for i in range(num_bands)]
# GLCM 특성 컬럼 이름 (각 특성별로 4방향: 0°, 90°, 180°, 270°)
glcm_props = ['GLCM_Contrast', 'GLCM_Energy', 'GLCM_Homogeneity', 'GLCM_Correlation']
# angles_deg = [0, 90, 180, 270]
angles_deg = [45, 135, 225, 315]
glcm_column_names = [f"{prop}_{ang}" for prop in glcm_props for ang in angles_deg]

column_names = band_columns + glcm_column_names
df = pd.DataFrame(combined_features, columns=column_names)

csv_path = "./src/gs_output_features.csv"
df.to_csv(csv_path, index=False)
print(f"CSV 저장 완료: {csv_path}")
print(f"CSV 저장 소요 시간: {time.time() - save_start_time:.2f}초")

total_time = time.time() - start_time
print(f"\n전체 처리 소요 시간: {total_time:.2f}초 ({total_time/60:.2f}분)")
