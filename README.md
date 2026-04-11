# 전기차 배터리 시계열 이상 탐지 with XAI

> **전기차 배터리팩의 셀 전압 시계열을 분석하여 이상을 탐지하고, XAI로 "왜 이상인가"까지 설명하는 다중 모델 비교 프로젝트**  
> _BOAZ 24기 분석 컨퍼런스_  
> _조유진 · 정명훈 · 김완철 · 이윤경_

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange) ![Keras Tuner](https://img.shields.io/badge/Keras--Tuner-Bayesian-red) ![SHAP](https://img.shields.io/badge/XAI-SHAP%20%7C%20Attention%20%7C%20GradCAM-green)

> _"이 셀, 지금 위험한가?"_ — 배터리 셀 전압의 미세한 패턴 변화는 화재·폭발의 전조가 될 수 있습니다. 단순히 분류하는 것을 넘어, **모델이 어느 셀·어느 시점을 보고 그렇게 판단했는지** 까지 해석하는 것이 본 프로젝트의 목표입니다.

---

## 1. 문제 정의: 왜 배터리 이상 탐지가 어려운가

전기차 배터리는 수십~수백 개의 셀이 직렬·병렬로 묶여 모듈을 이루고, 다시 모듈이 모여 팩을 구성합니다. 하나의 셀이라도 비정상 동작하면 전체 시스템 안전성에 영향을 줍니다.

<p align="center">
  <img src="assets/battery_structure.png" width="85%"/>
  <br/>
  <em>셀 → 모듈 → 팩 → 전기차로 이어지는 배터리 계층 구조</em>
</p>

| 일반 분류 문제 | 배터리 이상 탐지 |
|---|---|
| 클래스 간 경계가 시각적으로 명확 | **셀 전압의 미세한 시간적 변화**가 이상의 단서 |
| 한 시점만 봐도 분류 가능 | **연속된 시점의 흐름**이 있어야 의미 형성 |
| 오분류 비용이 균등 | **결함 미탐지(False Negative)의 비용이 압도적** |

특히 셀 전압이 정상 범위(3V ~ 4.2V)를 벗어나면 과방전·과충전 위험이 발생하고, 4.37V 이상에서는 화재로 직결될 수 있습니다.

<p align="center">
  <img src="assets/anomaly_criteria.png" width="85%"/>
  <br/>
  <em>차종(IONIQ 96셀 / NIRO·KONA 98셀)별 셀 구성과 비정상 전압 기준</em>
</p>

→ 단순 임계값 룰로는 점진적 열화를 잡지 못하므로, **시계열 패턴 학습 + 해석 가능성**을 모두 갖춘 파이프라인이 필요합니다.

## 2. Architecture 개요

전체 시스템은 **데이터 구축 파이프라인**(상)과 **모델링·해석 파이프라인**(하)으로 구성됩니다. 핵심 설계는 두 가지:

- **모델을 4종으로 분리**: Random Forest · XGBoost · LSTM · CNN을 동일 데이터로 학습시켜 성능과 해석력을 비교
- **각 모델에 맞는 XAI 기법 매칭**: 트리 계열은 SHAP, 시계열은 Attention, 이미지화는 Grad-CAM

```
NPY 셀 전압 → 폴더명 라벨링 → 96→98 셀 차원 통일 → 100×98 슬라이딩 윈도우
                                                          ↓
                                              ┌───────────┴───────────┐
                                              ↓                       ↓
                                       4-Model 병렬 학습          XAI 해석
                                  (RF / XGB / LSTM / CNN)   (SHAP / Attention / Grad-CAM)
```

| 담당 | 모델 | XAI 기법 |
| :--- | :--- | :--- |
| 김완철 | Random Forest | SHAP (Summary / Waterfall) |
| 조유진 | XGBoost | SHAP + Feature Importance |
| **정명훈** | **Bidirectional LSTM + Attention** | **Attention Weights 분석** |
| 이윤경 | CNN | SHAP / Grad-CAM |

> 본 README는 그중 **LSTM + Attention** 파트를 중심으로 작성되었습니다.

## 3. 데이터셋: AI-Hub 자율주행 고장진단

- **출처**: AI-Hub 자율주행 고장진단 데이터셋 (배터리 데이터)
- **차종**: IONIQ(96셀), NIRO·KONA(98셀)
- **라벨**: Normal / Caution / Defect (폴더명 기준 자동 라벨링)

<p align="center">
  <img src="assets/dataset_structure.png" width="90%"/>
  <br/>
  <em>차종·상태별 폴더 구조와 npy 파일 명명 규칙</em>
</p>

⚖️ 본 데이터는 AI-Hub 이용 약관에 따라 연구 목적으로만 활용되었습니다.

## 4. 데이터 전처리: Dynamic Window Construction

### 4-1. 96 → 98 셀 차원 통일
IONIQ는 96셀, NIRO/KONA는 98셀로 배터리 팩 구성이 달라 동일 모델에 입력 불가. IONIQ 데이터에 대해 행(row) 평균값 2개를 추가하여 98셀로 확장.

> ⚠️ **한계**: defect 샘플의 경우 정상에 가까운 평균값이 추가되어 약한 라벨 노이즈가 발생할 수 있음.

### 4-2. 슬라이딩 윈도우
상태별 npy 파일을 시간순으로 stack한 뒤, **100시점 × 98셀**의 윈도우로 슬라이딩하며 하나의 학습 패턴으로 인식. 최종적으로 **64,642개의 (100×98) 패턴**이 X_train으로 구축됨.

<p align="center">
  <img src="assets/sliding_window.png" width="75%"/>
  <br/>
  <em>시계열 셀 데이터 위를 100시점 윈도우로 슬라이딩하여 학습 패턴 생성</em>
</p>

→ 단일 시점이 아닌 **100시점의 흐름**을 하나의 입력으로 다루기 때문에, 점진적 열화나 미세한 패턴 변화도 학습 가능.

## 5. LSTM + Attention Implementation Details

단순한 시계열 분류를 넘어, **"왜 이상인가"** 를 설명할 수 있도록 Attention 기반으로 설계된 세부 로직입니다.

### 5-1. 모델 구조

<p align="center">
  <img src="assets/lstm_architecture.png" width="85%"/>
  <br/>
  <em>Bidirectional LSTM × N → Attention Layer → Dense Softmax</em>
</p>

```
Input (sequence_length=100, features=98)
   ↓
[ Bidirectional LSTM → Dropout ] × N    (N=2~4, Tuner 탐색)
   ↓
AttentionLayer  (가중치 계산 + weighted sum)
   ↓
Dense(3, softmax) → {Normal, Caution, Defect}
```

### 5-2. 왜 Bidirectional LSTM + Attention인가
- **Bi-LSTM**: 배터리 셀의 이상 신호는 과거뿐 아니라 회복 구간(미래 시점)을 함께 봐야 의미가 명확해짐 → 양방향 인코딩 필수
- **Attention Layer**: 100 time step 중 어느 시점이 분류에 결정적이었는지 가중치로 표현 → **해석 가능성** 확보 (SHAP 대체 수단으로도 활용)

### 5-3. 메모리 & 학습 최적화
- **Mixed Precision (`mixed_float16`)**: float16 연산으로 GPU 메모리 절반 절약, 연산 속도 향상
- **MinMaxScaler**: 3D 시계열을 2D로 reshape하여 정규화 후 원형 복원
- **Keras Tuner (Bayesian Optimization)**: LSTM 레이어 수(2~4), 유닛 수(64~256), Dropout(0.1~0.5), learning rate를 자동 탐색 (max_trials=10)
- **Exponential LR Decay**: `lr * 0.9^epoch` — 초반 빠른 수렴 + 후반 미세 조정
- **Early Stopping** (patience=10) 으로 과적합 방지
- L2 정규화 모델 vs Tuned-only 모델을 비교 후 더 나은 쪽을 최종 채택

## 6. 결과

<p align="center">
  <img src="assets/lstm_results.png" width="90%"/>
  <br/>
  <em>좌: Train/Val Loss 곡선 / 우: Final Classification Report (Test Acc 0.9658)</em>
</p>

| Class | Precision | Recall | F1-score | Support |
| :--- | :---: | :---: | :---: | :---: |
| Normal (0)  | 0.98 | 1.00 | 0.99 | 3,317 |
| Caution (1) | 0.96 | 0.96 | 0.96 | 2,408 |
| Defect (2)  | 0.95 | 0.93 | 0.94 | 2,376 |
| **Accuracy** | | | **0.9658** | 8,101 |

다만 Validation Loss가 epoch 후반에 상승하는 양상이 있어, 일반화 측면에서 추가 개선 여지가 존재합니다.

### 모델별 비교

| 모델 | Test Accuracy | XAI |
| :--- | :---: | :--- |
| Random Forest (n=500, depth=100) | 0.9875 | SHAP Summary / Waterfall |
| XGBoost (multi:softmax) | **0.9981** | SHAP + Feature Importance |
| **LSTM + Attention (Ours)** | **0.9658** | **Attention Weights** |
| CNN | 과적합 발생 | Grad-CAM |

## 7. XAI: Attention 분석 (SHAP 대체)

처음에는 SHAP의 KernelExplainer를 적용하려 했으나, **3D 시계열 입력에 대해 A100 GPU에서도 RAM OOM**이 발생해 포기했습니다. 대신 모델 내부의 Attention Layer가 출력하는 가중치를 직접 분석하는 방향으로 전환했습니다.

### 7-1. 핵심 발견: Time Step 61

모든 샘플에서 attention weight가 최대가 되는 시점은 **time step 61** (값 약 0.9992). 즉, 모델은 100시점 윈도우 중 특정 구간에 일관되게 집중하여 이상 여부를 판단합니다.

<p align="center">
  <img src="assets/attention_weights.png" width="75%"/>
  <br/>
  <em>Sample 0의 Attention Weights — 빨간 점은 weight ≥ 0.1인 중요 시점</em>
</p>

### 7-2. 원본 시계열과의 대응

같은 샘플의 정규화된 셀 전압 시계열을 보면, time step 40~60 구간에서 다수의 셀 값이 동시에 하강하는 패턴이 관찰됩니다. 모델은 바로 이 구간을 가장 중요한 의사결정 근거로 삼고 있음.

<p align="center">
  <img src="assets/raw_timeseries.png" width="75%"/>
  <br/>
  <em>Sample 0의 100 time step × 98 셀 정규화 전압 변화 — Attention이 강조한 구간과 시각적으로 일치</em>
</p>

→ 향후 이 그래프 위에 attention weight를 히트맵으로 오버레이하면 해석력이 더 강화될 것.

## 8. Conclusion

1. **시계열 패턴 학습** — Bi-LSTM + Attention으로 단일 시점 룰 기반 탐지의 한계를 넘어 96.58%의 분류 정확도 달성
2. **해석 가능한 이상 탐지** — Attention 가중치 분석으로 "어느 시점이 결정적이었는가"를 시각화, SHAP의 메모리 한계를 우회
3. **다중 모델 비교 프레임워크** — 동일 데이터셋 위에서 4가지 모델·XAI 조합을 비교해 모델별 강점과 한계를 정량적으로 파악

## 9. Future Direction

### 모델 측면
- **Feature Attention + Temporal Attention 2단 구조** — "어느 셀이 / 어느 시점에" 중요한지를 동시에 해석 가능
- **차종별 분리 학습** — 96→98 확장의 라벨 노이즈를 제거하기 위해 IONIQ / NIRO·KONA를 별도 모델로

### 해석 측면
- **메타데이터 컬럼 보존** — 각 샘플이 어느 날짜·파일·셀에 해당하는지 추적 가능한 컬럼을 학습에서는 제외하되 분석용으로 보관 → SHAP feature name에 활용
- **Attention × Raw Signal 오버레이** — 셀 전압 그래프 위에 attention weight를 색상으로 직접 표시

### 운영 측면
- **실시간 스트리밍 추론** — 슬라이딩 윈도우를 온라인으로 적용해 BMS에 직접 결합

---

## 🛠️ 기술 스택

**Modeling**: `TensorFlow / Keras` · `Bidirectional LSTM` · `Attention Layer`  
**Tuning**: `Keras Tuner (Bayesian Optimization)` · `Mixed Precision (float16)`  
**XAI**: `Attention Weights` · `SHAP` · `Grad-CAM`  
**Comparison Models**: `Random Forest` · `XGBoost` · `CNN`  
**Language**: `Python 3.10+`  
**Environment**: `Google Colab Pro (A100)`

## 📁 저장소 구조

```text
├── assets/                    # README 이미지
├── data/                      # AI-Hub 배터리 데이터 (gitignore)
│   ├── Dev_96/                # IONIQ
│   ├── Dev_98/                # KONA, NIRO
│   ├── X_train.npy            # (64642, 100, 98)
│   └── Y_train.npy
├── code/
│   ├── lstm_xai.ipynb         # 배터리_lstm복잡화.ipynb (LSTM + Attention)
│   ├── random_forest_xai.ipynb
│   ├── xgboost_xai.ipynb
│   └── cnn_xai.ipynb
├── models/                    # 학습된 가중치
│   ├── best_model_tuned.h5
│   └── lstm_tuner_attention_weights.npy
└── 발표자료.pdf
```

## 참여자

- **조유진**: XGBoost + XAI
- **정명훈**: LSTM + XAI — https://github.com/Cjapysql
- **김완철**: Random Forest + XAI
- **이윤경**: CNN + XAI

---
