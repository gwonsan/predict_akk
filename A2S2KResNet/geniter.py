import torch
import numpy as np
import torch.utils.data as Data

# 인덱스를 행과 열 위치로 변환하는 함수
def index_assignment(index, row, col, pad_length):
    new_assign = {}
    for counter, value in enumerate(index):
        assign_0 = value // col + pad_length  # 행 위치 계산 (padding 고려)
        assign_1 = value % col + pad_length   # 열 위치 계산 (padding 고려)
        new_assign[counter] = [assign_0, assign_1]
    return new_assign

# 주어진 위치를 중심으로 patch를 선택하는 함수
def select_patch(matrix, pos_row, pos_col, ex_len):
    selected_rows = matrix[range(pos_row-ex_len, pos_row+ex_len+1)]  # 행 선택
    selected_patch = selected_rows[:, range(pos_col-ex_len, pos_col+ex_len+1)]  # 열 선택
    return selected_patch


# 작은 큐빅 데이터를 선택하는 함수
def select_small_cubic(data_size, data_indices, whole_data, patch_length, padded_data, dimension):
    small_cubic_data = np.zeros((data_size, 2 * patch_length + 1, 2 * patch_length + 1, dimension),dtype=np.float32)
    print(small_cubic_data.shape)
    data_assign = index_assignment(data_indices, whole_data.shape[0], whole_data.shape[1], patch_length) # 인덱스 -> 위치
    print(len(data_assign))
    for i in range(len(data_assign)):
        small_cubic_data[i] = select_patch(padded_data, data_assign[i][0], data_assign[i][1], patch_length) # patch 선택
    return small_cubic_data


# 데이터 iteration을 생성하는 함수
def generate_iter(TRAIN_SIZE, train_indices, TEST_SIZE, test_indices, TOTAL_SIZE, total_indices, VAL_SIZE,
                  whole_data, PATCH_LENGTH, padded_data, INPUT_DIMENSION, batch_size, gt):
    gt_all = gt[total_indices] - 1 # 전체 데이터의 label
    y_train = gt[train_indices] - 1 # 훈련 데이터 label
    y_test = gt[test_indices] - 1   # 테스트 데이터 label

    all_data =  select_small_cubic(TOTAL_SIZE, total_indices, whole_data,
                                                      PATCH_LENGTH, padded_data, INPUT_DIMENSION) # 전체 데이터 cubic 선택

    train_data = select_small_cubic(TRAIN_SIZE, train_indices, whole_data,
                                                        PATCH_LENGTH, padded_data, INPUT_DIMENSION) # 훈련 데이터 cubic 선택
    print(train_data.shape)
    test_data =  select_small_cubic(TEST_SIZE, test_indices, whole_data,
                                                       PATCH_LENGTH, padded_data, INPUT_DIMENSION) # 테스트 데이터 cubic 선택
    x_train = train_data.reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2], INPUT_DIMENSION) # 훈련 데이터 reshape
    x_test_all = test_data.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2], INPUT_DIMENSION) # 전체 테스트 데이터 reshape

    x_val = x_test_all[-VAL_SIZE:] # validation data
    y_val = y_test[-VAL_SIZE:]   # validation label

    x_test = x_test_all[:-VAL_SIZE] # test data
    y_test = y_test[:-VAL_SIZE]   # test label
    
    # numpy array ->  torch tensor 변환 및 차원 추가
    x1_tensor_train = torch.from_numpy(x_train).type(torch.FloatTensor).unsqueeze(1)
    y1_tensor_train = torch.from_numpy(y_train).type(torch.FloatTensor)
    torch_dataset_train = Data.TensorDataset(x1_tensor_train, y1_tensor_train) # 훈련 데이터셋

    x1_tensor_valida = torch.from_numpy(x_val).type(torch.FloatTensor).unsqueeze(1)
    y1_tensor_valida = torch.from_numpy(y_val).type(torch.FloatTensor)
    torch_dataset_valida = Data.TensorDataset(x1_tensor_valida, y1_tensor_valida) # 검증 데이터셋

    x1_tensor_test = torch.from_numpy(x_test).type(torch.FloatTensor).unsqueeze(1)
    y1_tensor_test = torch.from_numpy(y_test).type(torch.FloatTensor)
    torch_dataset_test = Data.TensorDataset(x1_tensor_test,y1_tensor_test) # 테스트 데이터셋

    all_data.reshape(all_data.shape[0], all_data.shape[1], all_data.shape[2], INPUT_DIMENSION)
    all_tensor_data = torch.from_numpy(all_data).type(torch.FloatTensor).unsqueeze(1)
    all_tensor_data_label = torch.from_numpy(gt_all).type(torch.FloatTensor)
    torch_dataset_all = Data.TensorDataset(all_tensor_data, all_tensor_data_label) # 전체 데이터셋


    train_iter = Data.DataLoader(
        dataset=torch_dataset_train,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=True,  # 무작위 섞기
        num_workers=0,  # multi-processing number
    )
    valiada_iter = Data.DataLoader(
        dataset=torch_dataset_valida,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=True,  # 무작위 섞기
        num_workers=0,  # multi-processing number
    )
    test_iter = Data.DataLoader(
        dataset=torch_dataset_test,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=False,  # 순서대로
        num_workers=0,  # multi-processing number
    )
    all_iter = Data.DataLoader(
        dataset=torch_dataset_all,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=False,  # 순서대로
        num_workers=0,  # multi-processing number
    )
    return train_iter, valiada_iter, test_iter, all_iter #, y_test
