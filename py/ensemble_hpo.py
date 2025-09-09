# ensemble_hpo.py
import pandas as pd
import numpy as np
import warnings
from datetime import datetime
import os
import json

# 모델 클래스는 라이브러리에서 직접 import
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# 각 classifier.py 파일에서는 최적 파라미터(DEFAULT_PARAMS)만 import
from catboost_classifier import DEFAULT_PARAMS as cat_params_orig
from lightgbm_classifier import DEFAULT_PARAMS as lgb_params_orig
from xgboost_classifier import DEFAULT_PARAMS as xgb_params_orig

# 메타 모델 튜닝 및 OOF 생성을 위한 라이브러리 import
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, f1_score

warnings.filterwarnings('ignore')

# ===================================================================
#                      설정 변수
# ===================================================================
TRAIN_PATH = 'data/train.csv'
TEST_PATH = 'data/test.csv'  # OOF 생성 시 필요
N_SPLITS = 5
RANDOM_SEED = 42

PARAM_GRID = [
    {'penalty': ['l2', None], 'solver': ['lbfgs', 'newton-cg', 'sag', 'newton-cholesky'], 'C': [0.01, 0.1, 1, 10, 100]},
    {'penalty': ['l1', 'l2'], 'solver': ['liblinear', 'saga'], 'C': [0.01, 0.1, 1, 10, 100]},
    {'penalty': ['elasticnet'], 'solver': ['saga'], 'C': [0.01, 0.1, 1, 10, 100], 'l1_ratio': [0.3, 0.5, 0.7]}
]
# ===================================================================

# ===================================================================
#                      1차 모델(Base Models) 정의
# ===================================================================
cat_params = cat_params_orig.copy()
lgb_params = lgb_params_orig.copy()
xgb_params = xgb_params_orig.copy()

cat_params.pop('submission', None)
lgb_params.pop('submission', None)
xgb_params.pop('submission', None)

base_models = [
    ('CatBoost', CatBoostClassifier(**cat_params)),
    ('LightGBM', LGBMClassifier(**lgb_params)),
    ('XGBoost', XGBClassifier(**xgb_params))
]


# ===================================================================


def get_oof_predictions(X, y, X_test, models):
    print("Out-of-Fold 예측 피처를 생성합니다...")

    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    num_classes = len(y.unique())
    oof_train_preds = np.zeros((X.shape[0], len(models) * num_classes))

    # [수정] 원-핫 인코딩을 모델 루프 안으로 이동
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

    for model_idx, (name, model) in enumerate(models):
        print(f"\nProcessing Model: {name}", flush=True)

        X_model, X_test_model = X.copy(), X_test.copy()

        if name in ['LightGBM', 'XGBoost']:
            X_model = pd.get_dummies(X_model, columns=categorical_cols, drop_first=True)
            X_test_model = pd.get_dummies(X_test_model, columns=categorical_cols, drop_first=True)

            train_cols = X_model.columns
            missing_in_test = set(train_cols) - set(X_test_model.columns)
            for c in missing_in_test: X_test_model[c] = 0
            extra_in_test = set(X_test_model.columns) - set(train_cols)
            X_test_model = X_test_model.drop(columns=list(extra_in_test))
            X_test_model = X_test_model[train_cols]

        for fold_idx, (train_index, valid_index) in enumerate(kf.split(X_model, y)):
            print(f"  - Fold {fold_idx + 1}/{N_SPLITS} 학습 및 예측...", flush=True)
            X_train, X_valid = X_model.iloc[train_index], X_model.iloc[valid_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

            if name == 'CatBoost':
                cat_features = np.where(X_train.dtypes == 'object')[0]
                model.fit(X_train, y_train, cat_features=cat_features, eval_set=[(X_valid, y_valid)])
            elif name == 'XGBoost':
                model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
            else:
                model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])

            valid_preds = model.predict_proba(X_valid)
            oof_train_preds[valid_index, model_idx * num_classes:(model_idx + 1) * num_classes] = valid_preds

    print("\nOOF 피처 생성 완료.")
    return oof_train_preds


if __name__ == "__main__":
    print("데이터를 로드하고 전처리합니다...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)  # HPO 중에는 test_df가 직접 사용되지는 않음

    y_target = train_df['support_needs']
    train_df = train_df.drop(columns=['ID', 'support_needs'])
    test_df = test_df.drop(columns=['ID'])

    # [수정] 메인 부분의 전처리 로직 삭제 (get_oof_predictions 함수로 이동)

    oof_train = get_oof_predictions(train_df, y_target, test_df, base_models)

    print("\nGrid Search로 메타 모델 튜닝을 시작합니다...")

    f1_scorer = make_scorer(f1_score, average='macro')
    meta_model = LogisticRegression(random_state=RANDOM_SEED, max_iter=5000)

    grid_search = GridSearchCV(
        estimator=meta_model,
        param_grid=PARAM_GRID,
        scoring=f1_scorer,
        cv=5,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(oof_train, y_target)

    print("\n===== Meta Model HPO 완료 (Grid Search) =====")
    print(f"총 시도 횟수: {len(grid_search.cv_results_['params'])}")
    print(f"최고 점수 (f1_macro): {grid_search.best_score_:.4f}")
    print("최적 파라미터:")
    for key, value in grid_search.best_params_.items():
        print(f"  - {key}: {value}")