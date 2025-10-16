import collections
import math
import time
import os

import numpy as np
import scipy.io as sio
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics, preprocessing
from sklearn.decomposition import PCA
import pandas as pd

import geniter
import record
import Utils
from torchsummary import summary

PARAM_DATASET = 'GS'

PATCH_SIZE = 4

PARAM_KERNEL_SIZE = 24
# 불러올 모델 파일의 경로
MODEL_PATH = './models/2407_S3KAIResNetpatch_9_GS_split_0.9_lr_0.0001adam_kernel_240.991.pt' # <--- 여기에 모델 파일 경로를 정확하게 입력하세요.
# ======================================================================================

# # Data Loading
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

global Dataset
Dataset = PARAM_DATASET.upper()

def load_dataset(Dataset):
    """데이터셋을 로드하고 PCA를 적용하는 함수."""
    data_path = '../../DSNet/data/'
    if Dataset == 'IN':
        mat_data = sio.loadmat(data_path + 'Indian_pines_corrected.mat')
        mat_gt = sio.loadmat(data_path + 'Indian_pines_gt.mat')
        data_hsi = mat_data['indian_pines_corrected']
        gt_hsi = mat_gt['indian_pines_gt']
        K = 200
    elif Dataset == 'GS':
        image_file = '/home1/jmt30269/DSNet/data/202407_downsampled_gongsan.h5'
        with h5py.File(image_file, 'r') as f:
            data_hsi = f['input'][:]
            TR = f['TR'][:]
            TE = f['TE'][:]
            gt_hsi = TR + TE
        K = data_hsi.shape[2]
    elif Dataset == 'BS':
        image_file = '/home1/jmt30269/DSNet/data/202307_downsampled_buso.h5'
        with h5py.File(image_file, 'r') as f:
            data_hsi = f['input'][:]
            TR = f['TR'][:]
            TE = f['TE'][:]
            gt_hsi = TR + TE
        K = data_hsi.shape[2]
    else:
        raise ValueError("Unknown dataset")

    shapeor = data_hsi.shape
    # PCA 적용
    if K < data_hsi.shape[-1]:
        data_hsi = data_hsi.reshape(-1, data_hsi.shape[-1])
        pca = PCA(n_components=K)
        data_hsi = pca.fit_transform(data_hsi)
        shapeor = np.array(shapeor)
        shapeor[-1] = K
        data_hsi = data_hsi.reshape(shapeor)

    # 데이터셋의 총 레이블된 픽셀 수 계산
    TOTAL_SIZE = np.count_nonzero(gt_hsi)
    return data_hsi, gt_hsi, TOTAL_SIZE

# # Pytorch Data Loader Creation
data_hsi, gt_hsi, TOTAL_SIZE = load_dataset(Dataset)
print(f"Data shape: {data_hsi.shape}")
image_x, image_y, BAND = data_hsi.shape
data = data_hsi.reshape(np.prod(data_hsi.shape[:2]), np.prod(data_hsi.shape[2:]))
gt = gt_hsi.reshape(np.prod(gt_hsi.shape[:2]), )
CLASSES_NUM = np.max(gt)
print(f'The class numbers of the HSI data is: {CLASSES_NUM}')

PATCH_LENGTH = PATCH_SIZE
img_rows = 2 * PATCH_LENGTH + 1
img_cols = 2 * PATCH_LENGTH + 1
INPUT_DIMENSION = data_hsi.shape[2]

data = preprocessing.scale(data)
data_ = data.reshape(data_hsi.shape[0], data_hsi.shape[1], data_hsi.shape[2])
whole_data = data_
padded_data = np.pad(
    whole_data, ((PATCH_LENGTH, PATCH_LENGTH), (PATCH_LENGTH, PATCH_LENGTH), (0, 0)),
    'constant', constant_values=0)

# # Model Definition (기존 코드와 동일)
class eca_layer(nn.Module):
    """Constructs a ECA module."""
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv2d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w, t = x.size()
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -3)).transpose(-1, -3).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, use_1x1conv=False, stride=1, start_block=False, end_block=False):
        super(Residual, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.ReLU()
        )
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        if use_1x1conv:
            self.conv3 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        if not start_block:
            self.bn0 = nn.BatchNorm3d(in_channels)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)
        if start_block:
            self.bn2 = nn.BatchNorm3d(out_channels)
        if end_block:
            self.bn2 = nn.BatchNorm3d(out_channels)
        self.ecalayer = eca_layer(out_channels)
        self.start_block = start_block
        self.end_block = end_block

    def forward(self, X):
        identity = X
        if self.start_block:
            out = self.conv1(X)
        else:
            out = self.bn0(X)
            out = F.relu(out)
            out = self.conv1(out)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        if self.start_block:
            out = self.bn2(out)
        out = self.ecalayer(out)
        out += identity
        if self.end_block:
            out = self.bn2(out)
            out = F.relu(out)
        return out

class S3KAIResNet(nn.Module):
    def __init__(self, band, classes, reduction):
        super(S3KAIResNet, self).__init__()
        self.name = 'SSRN'
        self.conv1x1 = nn.Conv3d(in_channels=1, out_channels=PARAM_KERNEL_SIZE, kernel_size=(1, 1, 7), stride=(1, 1, 2), padding=0)
        self.conv3x3 = nn.Conv3d(in_channels=1, out_channels=PARAM_KERNEL_SIZE, kernel_size=(3, 3, 7), stride=(1, 1, 2), padding=(1, 1, 0))
        self.batch_norm1x1 = nn.Sequential(nn.BatchNorm3d(PARAM_KERNEL_SIZE, eps=0.001, momentum=0.1, affine=True), nn.ReLU(inplace=True))
        self.batch_norm3x3 = nn.Sequential(nn.BatchNorm3d(PARAM_KERNEL_SIZE, eps=0.001, momentum=0.1, affine=True), nn.ReLU(inplace=True))
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.conv_se = nn.Sequential(nn.Conv3d(PARAM_KERNEL_SIZE, band // reduction, 1, padding=0, bias=True), nn.ReLU(inplace=True))
        self.conv_ex = nn.Conv3d(band // reduction, PARAM_KERNEL_SIZE, 1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)
        self.res_net1 = Residual(PARAM_KERNEL_SIZE, PARAM_KERNEL_SIZE, (1, 1, 7), (0, 0, 3), start_block=True)
        self.res_net2 = Residual(PARAM_KERNEL_SIZE, PARAM_KERNEL_SIZE, (1, 1, 7), (0, 0, 3))
        self.res_net3 = Residual(PARAM_KERNEL_SIZE, PARAM_KERNEL_SIZE, (3, 3, 1), (1, 1, 0))
        self.res_net4 = Residual(PARAM_KERNEL_SIZE, PARAM_KERNEL_SIZE, (3, 3, 1), (1, 1, 0), end_block=True)
        kernel_3d = math.ceil((band - 6) / 2)
        self.conv2 = nn.Conv3d(in_channels=PARAM_KERNEL_SIZE, out_channels=128, padding=(0, 0, 0), kernel_size=(1, 1, kernel_3d), stride=(1, 1, 1))
        self.batch_norm2 = nn.Sequential(nn.BatchNorm3d(128, eps=0.001, momentum=0.1, affine=True), nn.ReLU(inplace=True))
        self.conv3 = nn.Conv3d(in_channels=1, out_channels=PARAM_KERNEL_SIZE, padding=(0, 0, 0), kernel_size=(3, 3, 128), stride=(1, 1, 1))
        self.batch_norm3 = nn.Sequential(nn.BatchNorm3d(PARAM_KERNEL_SIZE, eps=0.001, momentum=0.1, affine=True), nn.ReLU(inplace=True))
        self.avg_pooling = nn.AvgPool3d(kernel_size=(5, 5, 1))
        self.full_connection = nn.Sequential(nn.Linear(PARAM_KERNEL_SIZE, classes))

    def forward(self, X):
        x_1x1 = self.conv1x1(X)
        x_1x1 = self.batch_norm1x1(x_1x1).unsqueeze(dim=1)
        x_3x3 = self.conv3x3(X)
        x_3x3 = self.batch_norm3x3(x_3x3).unsqueeze(dim=1)
        x1 = torch.cat([x_3x3, x_1x1], dim=1)
        U = torch.sum(x1, dim=1)
        S = self.pool(U)
        Z = self.conv_se(S)
        attention_vector = torch.cat([self.conv_ex(Z).unsqueeze(dim=1), self.conv_ex(Z).unsqueeze(dim=1)], dim=1)
        attention_vector = self.softmax(attention_vector)
        V = (x1 * attention_vector).sum(dim=1)
        x2 = self.res_net1(V)
        x2 = self.res_net2(x2)
        x2 = self.batch_norm2(self.conv2(x2))
        x2 = x2.permute(0, 4, 2, 3, 1)
        x2 = self.batch_norm3(self.conv3(x2))
        x3 = self.res_net3(x2)
        x3 = self.res_net4(x3)
        x4 = self.avg_pooling(x3)
        x4 = x4.view(x4.size(0), -1)
        return self.full_connection(x4)

def get_all_labeled_indices(ground_truth):
    """레이블이 지정된 모든 픽셀의 인덱스를 반환하는 함수"""
    all_indices = []
    m = np.max(ground_truth)
    for i in range(1, m + 1):
        indices = [j for j, x in enumerate(ground_truth.ravel().tolist()) if x == i]
        all_indices.extend(indices)
    np.random.shuffle(all_indices)
    return all_indices

def sampling(proportion, ground_truth):
    train = {}
    test = {}
    labels_loc = {}
    m = max(ground_truth)
    for i in range(m):
        indexes = [
            j for j, x in enumerate(ground_truth.ravel().tolist())
            if x == i + 1
        ]
        np.random.shuffle(indexes)
        labels_loc[i] = indexes
        if proportion != 1:
            nb_val = max(int((1 - proportion) * len(indexes)), 3)
        else:
            nb_val = 0
        train[i] = indexes[:nb_val]
        test[i] = indexes[nb_val:]
    train_indexes = []
    test_indexes = []
    for i in range(m):
        train_indexes += train[i]
        test_indexes += test[i]
    np.random.shuffle(train_indexes)
    np.random.shuffle(test_indexes)
    return train_indexes, test_indexes

# # Inference (Test)
print("----------- Starting Inference -----------")

# 모델 초기화
net = S3KAIResNet(BAND, int(CLASSES_NUM), 2).to(device)
summary(net, (1, img_rows, img_cols, BAND))

# 학습된 모델 로드
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file not found at {MODEL_PATH}")
    exit()

print(f"Loading model from: {MODEL_PATH}")
net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
net.eval() # 추론 모드로 설정

# 전체 데이터셋의 인덱스 생성
all_labeled_indices = get_all_labeled_indices(gt)
TEST_SIZE = len(all_labeled_indices)
print(f'\nUsing the entire labeled dataset for testing.')
print(f'Test size (total labeled pixels): {TEST_SIZE}')
VALIDATION_SPLIT=0.9
# 데이터 이터레이터 생성
train_indices, test_indices = sampling(VALIDATION_SPLIT, gt)
_, total_indices = sampling(1, gt)

TRAIN_SIZE = len(train_indices)
print('Train size: ', TRAIN_SIZE)
TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE
print('Test size: ', TEST_SIZE)
VAL_SIZE = int(TRAIN_SIZE)
print('Validation size: ', VAL_SIZE)

print('-----Selecting Small Pieces from the Original Cube Data-----')
train_iter, valida_iter, test_iter, all_iter = geniter.generate_iter(
    TRAIN_SIZE, train_indices, TEST_SIZE, test_indices, TOTAL_SIZE,
    total_indices, VAL_SIZE, whole_data, PATCH_LENGTH, padded_data,
    INPUT_DIMENSION, 16, gt)  #batchsize in 1

# 1. 성능 평가 (전체 데이터셋)
print("\nCalculating performance metrics on the ENTIRE labeled dataset...")
pred_test = []
tic = time.time()
with torch.no_grad():
    for X, y in test_iter:
        X = X.to(device)
        y_hat = net(X)
        pred_test.extend(np.array(y_hat.cpu().argmax(axis=1)))
toc = time.time()

# GT (Ground Truth) 준비
gt_test = gt[all_labeled_indices] - 1
if len(pred_test) != len(gt_test):
    gt_test = gt_test[:len(pred_test)]

# --- 평가 지표 계산 ---
overall_acc = metrics.accuracy_score(gt_test, pred_test)
confusion_matrix = metrics.confusion_matrix(gt_test, pred_test)
each_acc, average_acc = record.aa_and_each_accuracy(confusion_matrix)
kappa = metrics.cohen_kappa_score(gt_test, pred_test)

# Precision, Recall, F1-score (Macro Average)
# zero_division=0: 특정 클래스로 예측된 샘플이 없을 경우, 해당 클래스의 precision/f1 score를 0으로 처리
precision_macro = metrics.precision_score(gt_test, pred_test, average='macro', zero_division=0)
recall_macro = metrics.recall_score(gt_test, pred_test, average='macro', zero_division=0)
f1_macro = metrics.f1_score(gt_test, pred_test, average='macro', zero_division=0)

# 클래스별 Precision, Recall, F1-score
precision_each = metrics.precision_score(gt_test, pred_test, average=None, zero_division=0)
recall_each = metrics.recall_score(gt_test, pred_test, average=None, zero_division=0)
f1_each = metrics.f1_score(gt_test, pred_test, average=None, zero_division=0)

TESTING_TIME = toc - tic

# --- 평가 결과 출력 ---
print(f"\n------ Test Performance Metrics (Full Dataset) ------")
print(f"Overall Accuracy (OA): {overall_acc:.4f}")
print(f"Average Accuracy (AA): {average_acc:.4f}")
print(f"Kappa Score          : {kappa:.4f}")
print(f"Precision (Macro)    : {precision_macro:.4f}")
print(f"Recall (Macro)       : {recall_macro:.4f}")
print(f"F1-score (Macro)     : {f1_macro:.4f}")
print(f"Testing Time         : {TESTING_TIME:.2f} seconds")
print("-----------------------------------------------------")

print("\n------ Per-Class Performance ------")
print(f"{'Class':>5} | {'Precision':>10} | {'Recall':>10} | {'F1-score':>10}")
print("-" * 43)
for i in range(len(precision_each)):
    print(f"{i+1:>5} | {precision_each[i]:>10.3f} | {recall_each[i]:>10.3f} | {f1_each[i]:>10.3f}")
print("-" * 43)


# 2. 전체 데이터에 대한 예측 및 결과 저장 (확률값 포함)
print("\nGenerating classification map, probabilities, and CSV file...")

predictions = []
probabilities = [] # 각 클래스 확률을 저장할 리스트 추가

# all_iter는 test_iter와 동일한 데이터를 가리키지만, 명확성을 위해 그대로 사용
with torch.no_grad():
    for X, _ in all_iter:
        X = X.to(device)
        
        # 모델의 출력(logits)을 얻음
        outputs = net(X)
        
        # Softmax를 적용하여 확률 계산
        # dim=1은 클래스 차원에 대해 소프트맥스를 적용하라는 의미
        probs = F.softmax(outputs, dim=1).cpu().numpy()
        probabilities.extend(probs)
        
        # 가장 높은 확률을 가진 클래스의 인덱스를 예측값으로 선택
        pred = np.argmax(probs, axis=1)
        predictions.extend(pred)

# --- 예측 결과와 확률을 CSV로 저장 ---
os.makedirs('./classification_results', exist_ok=True)

# 예측된 레이블은 0부터 시작하므로, 실제 클래스 번호(1부터 시작)에 맞추기 위해 +1을 해줍니다.
# gt_test를 만들 때 -1을 했던 것과 반대 과정입니다.
predicted_classes_1_based = [p + 1 for p in predictions]

# 확률값을 담을 데이터프레임 생성
prob_df = pd.DataFrame(probabilities)
# 각 컬럼에 클래스별 확률임을 명시하는 이름 부여
prob_df.columns = [f'class_{i+1}_prob' for i in range(CLASSES_NUM)]

# 최종적으로 저장할 데이터프레임 생성
# all_labeled_indices의 길이를 예측된 샘플 수에 맞게 슬라이싱
num_predictions = len(predictions)
final_df = pd.DataFrame({
    'index': all_labeled_indices[:num_predictions],
    'predicted_class': predicted_classes_1_based
})

# 인덱스, 예측 클래스 DataFrame과 확률 DataFrame을 옆으로 합치기
final_df = pd.concat([final_df, prob_df], axis=1)

# CSV 파일로 저장
csv_filename = os.path.basename(MODEL_PATH).replace('.pt', '_origin_fulldata_with_probs.csv')
csv_path = f'./classification_results/{csv_filename}'
final_df.to_csv(csv_path, index=False)
print(f"Classification results with probabilities saved to CSV: {csv_path}")


# --- 분류 결과 시각화 ---
os.makedirs('./classification_maps', exist_ok=True)
png_filename = os.path.basename(MODEL_PATH).replace('.pt', '_origin_fulldata2_map')
Utils.generate_png(
    all_iter, net, gt_hsi, Dataset, device, all_labeled_indices,
    f'./classification_maps/{png_filename}')

print("Classification map image saved in 'classification_maps' folder.")
print("\n----------- Inference Complete -----------")