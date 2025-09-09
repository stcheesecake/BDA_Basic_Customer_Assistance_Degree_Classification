import pandas as pd
import numpy as np
import os
from datetime import datetime
from itertools import combinations

# ===================================================================
#                      사용자 설정 변수
# ===================================================================
# 원본 데이터 경로 (수정하지 않음)
BASE_FEATURED_DATASET = 'data/new_train.csv'

# 실험할 모델 이름 (수정하지 않음)
MODEL = 'xgboost'  # 'catboost', 'xgboost' 등으로 변경하여 사용

# 실험을 반복할 시드(seed) 목록
SEEDS = [42, 43, 44, 45, 46]

# 실험할 새로운 피처 후보 전체 목록입니다.
ALL_NEW_FEATURES = [
    'is_older_group', 'older_and_member', 'is_low_frequency',
    'vip_inactive', 'new_inactive'
]
# ===================================================================

# [추가] MODEL 변수값에 따라 실제 사용할 모델 스크립트를 동적으로 import
if MODEL == 'lightgbm':
    import lightgbm_classifier as model_module
elif MODEL == 'catboost':
    import catboost_classifier as model_module
elif MODEL == 'xgboost':
    import xgboost_classifier as model_module
else:
    raise ValueError("지원하지 않는 모델입니다")


def run_and_log_experiment(model_module, dataset_path, features_to_include, seeds, log_file_path):
    """
    지정된 피처 조합으로 실험을 수행하고 결과를 CSV 파일에 기록합니다.
    """
    if not features_to_include:
        included_features_str = 'Original_Baseline'
    else:
        included_features_str = ', '.join(features_to_include)

    print("\n" + "=" * 50)
    print(f"실험 시작: 포함된 피처 = {included_features_str}")
    print("=" * 50)

    f1_scores, accuracy_scores = [], []

    full_df = pd.read_csv(dataset_path)

    features_to_exclude = [f for f in ALL_NEW_FEATURES if f not in features_to_include]
    existing_features_to_drop = [col for col in features_to_exclude if col in full_df.columns]
    df = full_df.drop(columns=existing_features_to_drop)

    temp_train_path = 'temp_train_for_experiment.csv'
    df.to_csv(temp_train_path, index=False)

    for seed in seeds:
        # [수정] MODEL 변수 대신, import된 model_module을 사용
        result = model_module.train_and_eval(
            train_path=temp_train_path,
            seed=seed,
            produce_artifacts=False,
        )
        f1_scores.append(result['metrics']['f1_macro'])
        accuracy_scores.append(result['metrics']['accuracy'])

    os.remove(temp_train_path)

    mean_f1 = np.mean(f1_scores)
    mean_accuracy = np.mean(accuracy_scores)

    print(f"-> 평균 F1-Macro: {mean_f1:.4f}, 평균 Accuracy: {mean_accuracy:.4f}")

    log_results(included_features_str, mean_f1, mean_accuracy, log_file_path)


def log_results(included_features_str, f1_macro, accuracy, log_file):
    """
    실험 결과를 CSV 파일에 한 줄 추가합니다.
    """
    new_log = pd.DataFrame({
        'TIMESTAMP': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        '포함된 FEATURE': [included_features_str],
        'F1 MACRO': [f"{f1_macro:.4f}"],
        'ACCURACY': [f"{accuracy:.4f}"]
    })

    if not os.path.exists(log_file):
        new_log.to_csv(log_file, index=False, encoding='utf-8-sig')
    else:
        new_log.to_csv(log_file, mode='a', header=False, index=False, encoding='utf-8-sig')


# --- 메인 실행 부분 ---
if __name__ == "__main__":
    output_dir = 'results/eda/feature_engineering'
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
    log_file_path = os.path.join(output_dir, f"{timestamp}_{MODEL}_feature_experiments.csv")
    print(f"실험 결과는 다음 파일에 저장됩니다: {log_file_path}")

    all_combinations = []
    all_combinations.append([])
    for r in range(1, len(ALL_NEW_FEATURES) + 1):
        for combo in combinations(ALL_NEW_FEATURES, r):
            all_combinations.append(list(combo))

    print(f"총 {len(all_combinations)}개의 피처 조합에 대한 실험을 시작합니다.")

    for features in all_combinations:
        run_and_log_experiment(
            model_module=model_module,  # [수정] 사용할 모델 모듈 전달
            dataset_path=BASE_FEATURED_DATASET,
            features_to_include=features,
            seeds=SEEDS,
            log_file_path=log_file_path
        )

    print("\n\n===== 모든 피처 조합 실험 완료 =====")
    print(f"전체 결과는 '{log_file_path}'를 확인하세요.")