# 고객 지원 필요 분류 모델 
## 1. 프로젝트 개요

본 프로젝트는 고객의 기본 정보 및 행동 데이터를 활용하여  
**고객이 어느 정도의 지원(Assistance / Support)이 필요한지를 사전에 예측하는 분류 모델**을 개발하는 것을 목표로 합니다.

고객 지원 필요 수준을 사전에 예측을 통한 기대효과는 아래와 같습니다.
- 상담 리소스를 효율적으로 배분
- 고위험 고객을 우선적으로 대응
- 고객 이탈 및 불만을 사전에 완화

## 2. 문제 정의

### 2.1 목표
고객 데이터 기반 **고객 지원 필요 수준 (Support_needs)** 을 예측하는 다중 분류(Classification) 문제를 해결합니다.

- 입력: 고객의 수치형 / 범주형 특성
- 출력: 고객 지원 필요 수준 (Class Label)

### 2.2 문제의 어려움
- **Class Imbalance**  
  → 특정 지원 수준에 데이터가 편중되어 있어 단순 정확도 기준 모델은 성능 왜곡 가능
- **Tabular 데이터 특성**  
  → 피처 간 비선형 관계 및 상호작용이 중요

## 3. 데이터 특성 분석 (EDA)

### 3.1 데이터 구조
- 대부분 **수치형(Numerical)** 및 **범주형(Categorical)** 중심의 Tabular 데이터
- 텍스트 기반 정보는 제한적이거나 없음
- 학습/테스트 데이터 분리 제공

### 3.2 클래스 불균형
- Support_needs 라벨 분포가 불균형
- Accuracy 대신 **F1 Score**를 주요 평가 지표로 사용

### 3.3 Feature 영향도 분석
각 Feature가 Support_needs에 미치는 영향을 정량·정성적으로 분석

사용 기법:
- **t-test / ANOVA**
- **KDE Plot**
- 클래스별 분포 비교

이를 통해
- 의미 없는 Feature 제거
- 중요 Feature 우선 활용


## 4. 접근 방법 및 실험 전략

### 4.1 Feature Engineering 전략
- 무작위 조합보다는 **EDA 기반 선택적 파생 변수 생성**
- 불필요한 Feature 증식을 피하고, 일반화 성능을 우선

### 4.2 AutoML 방식 도입 시도 (결론: 미채택)
- Feature 자동 조합 기반 AutoML 스타일 접근을 실험
- 일부 학습 데이터에서는 성능 개선
- 그러나:
  - 검증 성능 불안정
  - 과적합 위험 증가
- **Tabular 데이터 특성상 명시적 Feature 설계가 더 효과적이라 판단하여 최종 제외**


## 5. 모델링

Tabular 데이터에 강점을 가지는 Tree 계열 모델을 중심으로 다양한 모델을 비교 실험

### 실험 모델
- **CatBoost**
- **LightGBM**
- **XGBoost**
- **FT-Transformer**
- **TabNet**

### 모델 선택 기준
- Class Imbalance 상황에서의 F1 Score
- 검증 성능의 안정성
- 과적합 여부


## 6. 실험 설계 및 코드 구조

본 프로젝트는 EDA → Feature Engineering → 모델 학습 → 평가의 전체 파이프라인이
명확히 분리된 구조로 설계되었습니다.
실험 과정에서 발생할 수 있는 
- 실험 재현성 문제 
- EDA 코드와 학습 코드의 결합 
- 실험 결과 관리의 복잡성

을 최소화하는 것을 목표로 구조를 설계했습니다.

### 6.1 프로젝트 구조
```text
BDA_Basic_Customer_Assistance_Degree_Classification
├── data
│   ├── raw
│   │   ├── train.csv
│   │   └── test.csv
│   └── processed
│       ├── train_processed.csv
│       └── test_processed.csv
│
├── eda
│   ├── eda_basic.py
│   ├── eda_distribution.py
│   ├── eda_correlation.py
│   └── results
│       ├── describe.csv
│       ├── corr.csv
│       └── kde_plots
│
├── features
│   ├── feature_engineering.py
│   └── feature_selection.py
│
├── models
│   ├── train_catboost.py
│   ├── train_lightgbm.py
│   ├── train_xgboost.py
│   ├── train_tabnet.py
│   └── train_ft_transformer.py
│
├── evaluation
│   ├── metrics.py
│   ├── confusion_matrix.py
│   └── f1_analysis.py
│
├── utils
│   ├── seed.py
│   ├── logger.py
│   └── config.py
│
├── notebooks
│   └── exploration.ipynb
│
├── requirements.txt
└── README.md
```


## 7. 학습 및 검증 전략
### 7.1 데이터 분할
- train_test_split
- Stratified Sampling 적용
- 학습 : 검증 = 80 : 20

### 7.2 평가 지표
- Macro F1 Score를 주요 지표로 사용
- Class Imbalance 상황에서의 성능 왜곡 방지 목적
- 보조 지표:
  - Confusion Matrix
  - Class-wise Recall / Precision


## 8. 실험 결과 요약
### 8.1 모델별 성능 비교
- Tree 계열 모델(CatBoost, LightGBM)이 전반적으로 안정적인 성능
-  Deep Tabular 모델(FT-Transformer, TabNet)은 충분한 튜닝 시 경쟁력 있으나 데이터 규모 대비 과적합 위험 존재

### 9.2 최종 선택 모델
- CatBoost
    - 범주형 처리의 안정성
    - 비교적 적은 튜닝으로도 높은 F1 Score
    - 검증 성능 변동성 낮음

<br>


## 10. 실행 방법
- 현재 최종으로는 catboost_classifier.py 실행으로 문서상의 결과를 얻을 수 있음.
