import pandas as pd
import numpy as np
import os
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from catboost_classifier import DEFAULT_PARAMS as cat_params_orig
from lightgbm_classifier import DEFAULT_PARAMS as lgb_params_orig
from xgboost_classifier import DEFAULT_PARAMS as xgb_params_orig

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score

# ===================================================================
#                      설정 변수 (사용자 설정 유지)
# ===================================================================
TRAIN_PATH = 'data/train.csv'
TEST_PATH = 'data/test.csv'
N_SPLITS = 5
RANDOM_SEED = 42

# 모델별로 사용할 피처를 선택하기 위한 리스트
FEATURE_TOTAL = [
    'is_older_group', 'older_and_member', 'is_low_frequency',
    'vip_inactive', 'new_inactive'
]
FEATURE_CATBOOST = ['is_low_frequency', 'vip_inactive']
FEATURE_LIGHTGBM = []
FEATURE_XGBOOST = ['older_and_member', 'vip_inactive', 'new_inactive']
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
    oof_test_preds = np.zeros((X_test.shape[0], len(models) * num_classes))

    original_features = [col for col in X.columns if col not in FEATURE_TOTAL]

    for model_idx, (name, model) in enumerate(models):
        print(f"\nProcessing Model: {name}", flush=True)

        if name == 'CatBoost':
            model_features = FEATURE_CATBOOST
        elif name == 'LightGBM':
            model_features = FEATURE_LIGHTGBM
        elif name == 'XGBoost':
            model_features = FEATURE_XGBOOST

        final_feature_list = original_features + model_features
        X_subset = X[final_feature_list]
        X_test_subset = X_test[final_feature_list]
        print(f"  -> Using {len(final_feature_list)} features.")

        X_model, X_test_model = X_subset.copy(), X_test_subset.copy()
        categorical_cols = X_model.select_dtypes(include=['object']).columns.tolist()

        if name in ['LightGBM', 'XGBoost']:
            X_model = pd.get_dummies(X_model, columns=categorical_cols, drop_first=True)
            X_test_model = pd.get_dummies(X_test_model, columns=categorical_cols, drop_first=True)

            train_cols = X_model.columns
            missing_in_test = set(train_cols) - set(X_test_model.columns)
            for c in missing_in_test: X_test_model[c] = 0
            extra_in_test = set(X_test_model.columns) - set(train_cols)
            X_test_model = X_test_model.drop(columns=list(extra_in_test))
            X_test_model = X_test_model[train_cols]

        test_preds_per_fold = np.zeros((X_test_model.shape[0], num_classes))

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
            test_preds_per_fold += model.predict_proba(X_test_model)

        oof_test_preds[:, model_idx * num_classes:(model_idx + 1) * num_classes] = test_preds_per_fold / N_SPLITS

    print("\nOOF 피처 생성 완료.")
    return oof_train_preds, oof_test_preds


if __name__ == "__main__":
    # 1. 데이터 로드 및 전처리
    print("데이터를 로드하고 전처리합니다...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    test_ids = test_df['ID']

    y_target = train_df['support_needs']

    # 원본 피처와 신규 피처가 모두 포함된 데이터프레임 준비
    # (ID와 타겟 컬럼만 제거)
    train_df_features = train_df.drop(columns=['ID', 'support_needs'])

    # test_df는 원본 피처만 가지고 있으므로, train_df에서 신규 피처들을 가져와 추가해야 함
    # 이 과정은 feature_engineering.py에서 test.csv에 대해서도 수행되었다고 가정
    # 만약 아니라면, 아래 코드는 test_df에 신규 피처가 없어서 오류 발생 가능
    test_df_features = test_df.drop(columns=['ID'])

    oof_train, oof_test = get_oof_predictions(train_df_features, y_target, test_df_features, base_models)

    # ... (이하 메타 모델 학습 및 최종 예측 로직은 동일) ...
    print("\n메타 모델을 학습합니다...")
    meta_model = LogisticRegression(random_state=RANDOM_SEED, n_jobs=-1)
    meta_model.fit(oof_train, y_target)
    print("메타 모델 학습 완료.")

    print("\n앙상블 모델의 최종 성능을 평가합니다...")
    ensemble_oof_preds = meta_model.predict(oof_train)
    ensemble_f1 = f1_score(y_target, ensemble_oof_preds, average="macro")
    ensemble_accuracy = accuracy_score(y_target, ensemble_oof_preds)

    print("\n===== 최종 앙상블 모델 성능 (OOF 기준) =====")
    print(f"Accuracy: {ensemble_accuracy:.4f}")
    print(f"F1-Macro: {ensemble_f1:.4f}")
    print("=" * 45)

    print("\n최종 예측을 수행하고 제출 파일을 생성합니다...")
    final_predictions = meta_model.predict(oof_test)
    submission = pd.DataFrame({'ID': test_ids, 'support_needs': final_predictions})

    output_dir = 'results/ensemble'
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
    submission_filename = f"{timestamp}_submission.csv"
    submission_path = os.path.join(output_dir, submission_filename)
    submission.to_csv(submission_path, index=False)
    print(f"최종 제출 파일이 '{submission_path}'에 저장되었습니다.")