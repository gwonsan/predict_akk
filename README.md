# HSI 분류 모델 및 실험

이 저장소는 초분광 영상(HSI) 분류를 위한 코드 모음입니다. 다양한 딥러닝 모델, 전처리 스크립트 및 실험 결과를 포함합니다.

## 디렉토리 구조

*   ### `A2S2KResNet/`
    이 디렉토리는 HSI 분류를 위한 딥러닝 아키텍처인 **A2S2K-ResNet** 모델의 구현을 포함합니다.
    *   `A2S2KResNet.py`: 모델 아키텍처, 학습 루프 및 평가를 정의하는 메인 스크립트입니다.
    *   `run.sh`, `run2.sh`: 학습 및 평가를 실행하기 위한 셸 스크립트입니다.
    *   `models/`: 학습된 모델 체크포인트를 저장하는 디렉토리입니다.
    *   `report/`: 성능 리포트를 저장하는 디렉토리입니다.
    *   `classification_results/`: 분류 맵과 결과를 저장하는 디렉토리입니다.

*   ### `dsnet/`
    이 디렉토리는 HSI 분류를 위한 또 다른 딥러닝 모델인 **DSNet**(Dual-branch Subpixel-guided Network)의 구현을 포함합니다.
    *   `dsnet.ipynb`: DSNet 모델의 아키텍처와 학습 과정을 설명하는 Jupyter Notebook입니다.
    *   `demo.ipynb`: 모델 사용법을 보여주는 데모 Jupyter Notebook입니다.

*   ### `GAHT/`
    이 디렉토리는 HSI 분류 모델을 학습하고 평가하기 위한 일반적인 프레임워크를 제공합니다.
    *   `main.ipynb`: 실험을 실행하고 모델, 데이터셋 및 기타 하이퍼파라미터를 지정할 수 있는 메인 Jupyter Notebook입니다.
    *   `resolution_test.ipynb`: 해상도 테스트 관련 실험을 담은 Jupyter Notebook입니다.

*   ### `preprocess/`
    이 디렉토리는 분류 모델을 위한 특징을 생성하는 데 사용할 수 있는 데이터 전처리 스크립트를 포함합니다.
    *   `preprocessing_model_xgboost.ipynb`: XGBoost 모델을 위한 데이터 전처리 및 특징 공학을 다루는 Jupyter Notebook입니다.

*   ### `xgb/`
    이 디렉토리는 HSI 분류를 위해 **XGBoost** 알고리즘을 사용하는 실험 전용입니다. 분석 및 비교를 위한 Jupyter Notebook을 포함합니다.
    *   `논문용.ipynb`: 연구 논문을 위한 결과를 생성하는 Notebook입니다.
    *   `compare_dist.ipynb`: 분포를 비교하는 Notebook입니다.
    *   `shapval.ipynb`: SHAP(SHapley Additive exPlanations) 값 분석을 위한 Notebook입니다.

## 실행 방법

*   **A2S2KResNet**: 제공된 셸 스크립트를 사용하여 모델을 실행할 수 있습니다.
    ```bash
    cd A2S2KResNet
    sh run.sh
    ```
    또는 인자와 함께 python 스크립트를 직접 실행할 수 있습니다.
    ```bash
    python A2S2KResNet.py --dataset GS --epoch 100 --patch 4
    ```

*   **GAHT**, **dsnet**, **preprocess**, **xgb**: 각 디렉토리의 Jupyter Notebook (`.ipynb`) 파일을 열어 셀을 순서대로 실행하면 됩니다.

## 의존성

이 저장소의 프로젝트는 여러 Python 라이브러리에 의존합니다. 주요 의존성은 다음과 같습니다.

*   PyTorch
*   scikit-learn
*   NumPy
*   Pandas
*   h5py
*   rasterio
*   scikit-image
*   wandb (`A2S2KResNet`용)
*   torch-optimizer (`A2S2KResNet`용)
*   matplotlib
*   jupyter

pip를 사용하여 의존성을 설치할 수 있습니다.
```bash
pip install torch torchvision torchaudio
pip install scikit-learn numpy pandas h5py rasterio scikit-image wandb torch-optimizer matplotlib jupyter
```