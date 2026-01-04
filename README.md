# 🖼️ CIFAR Classification & Imbalanced Learning (ResNet & Mixup)

![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python&logoColor=white)
![ResNet](https://img.shields.io/badge/Model-ResNet20-success?style=flat-square)

## 📖 Project Overview
이 프로젝트는 **CIFAR-100** 및 **CIFAR-10** 데이터셋을 활용하여 CNN 모델 구조 비교와 데이터 불균형 문제를 해결하는 딥러닝 연구 프로젝트입니다.
**ResNet20**을 직접 구현하여 PlainNet과의 성능 차이를 분석하고, 전이 학습(Transfer Learning) 실험을 수행했습니다. 또한, 클래스 불균형 상황에서 **Balanced Mixup** 기법을 제안하여 베이스라인 대비 성능을 대폭 향상시켰습니다.

> **💡 Key Objective:** ResNet의 Residual Block 효과를 검증하고, Long-tailed Dataset(불균형 데이터)에서 일반화 성능을 높이는 것.

<br/>

## 📂 Repository Structure
본 프로젝트는 실험 보고서와 구현 코드를 체계적으로 관리하기 위해 다음과 같이 구성되었습니다.

```bash
cifar-resnet-imbalanced-learning/
│
├── 📂 docs/                  # 프로젝트 실험 보고서 (PDF)
├── 📂 notebooks/             # 모델 구현 및 학습 코드 (.ipynb)
├── .gitignore             # Git 추적 제외 설정
├── README.md              # 프로젝트 메인 문서
└── requirements.txt       # 의존성 라이브러리 목록
```

<br/>

## 📑 Project Report
모델 구조에 대한 이론적 배경, 실험 설계, 그리고 실험별 Loss/Accuracy 그래프는 PDF 보고서에 상세히 기술되어 있습니다.

* 📄 **Detailed Report:** [👉 프로젝트 결과 보고서 (PDF) 다운로드](./docs/CIFAR_Imbalanced_Learning_Report.pdf)

<br/>

## 🛠️ Methodology & Experiments

### 1. Model Architecture Comparison (PlainNet vs ResNet)
* **실험:** 동일한 깊이(20 layers)와 파라미터 수를 가진 PlainNet과 ResNet을 구현하여 비교.
* **결과:** **Skip Connection(Residual Block)**이 기울기 소실(Vanishing Gradient) 문제를 해결하여, ResNet이 PlainNet보다 학습 수렴 속도가 빠르고 Test Accuracy가 **약 10%p 이상 높음**을 확인했습니다.

### 2. Transfer Learning Strategy
* **실험:** ImageNet으로 사전 학습된(Pretrained) ResNet을 CIFAR-100에 맞게 미세 조정(Fine-tuning).
* **결과:** 마지막 FC Layer만 학습하는 것보다, **전체 모델을 미세 조정(Full Fine-tuning)**했을 때 성능이 월등히 우수함을 입증했습니다.

### 3. Handling Imbalanced Dataset (Long-tailed)
* **문제:** 특정 클래스의 데이터가 부족한 불균형 상황(Imbalanced CIFAR-10)에서 소수 클래스 예측 성능이 저하됨.
* **해결책 (Proposed Method):**
  * **Balanced Mixup:** 다수 클래스와 소수 클래스 데이터를 섞어(Mix) 샘플링하는 기법 적용.
  * **Balanced Softmax Loss:** 클래스 빈도에 따라 Loss 계산 시 마진(Margin)을 조정.
* **성과:** 베이스라인(Standard CE Loss) 대비 **F1 Score가 5%p 이상 향상**되었습니다.

<br/>

## 📈 Performance Summary (Imbalanced Task)

| Model | Technique | F1 Score | Improvement |
|:---:|:---|:---:|:---:|
| **Baseline** | Standard CrossEntropy | 0.7457 | - |
| **Proposed** | **Balanced Mixup** | **0.79xx** | **+5.0%p** 🏆 |

<br/>

## 🚀 Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run Experiments
```bash
# Jupyter Notebook 실행
jupyter notebook notebooks/cifar_classification_resnet.ipynb
```

<br/>

### 👤 Author
Github: @PaleBlueNote
Email: yoonseokchan0731@gmail.com

---
*This project was conducted to study Deep Learning architectures and Imbalanced Learning strategies.*