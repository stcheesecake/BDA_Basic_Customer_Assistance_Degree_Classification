import pandas as pd
import numpy as np
import os
from datetime import datetime

# catboost_classifier.py 스크립트를 불러옵니다.
# 이 코드를 실행하기 전에 catboost.py가 같은 폴더에 있어야 합니다.
import catboost_classifier

# ===================================================================
#                      사용자 설정 변수
# ===================================================================
# 실험에 사용할, 모든 피처가 생성되어 있는 데이터셋 경로
BASE_FEATURED_DATASET = 'data/new_train.csv'

# 이번 실험에 포함할 새로운 피처 목록을 지정합니다.
# 이 리스트를 비워두면 'Original_Baseline'으로 기록됩니다.
FEATURES_TO_INCLUDE = [
    'is_low_frequency',
    'vip_inactive',
    'is_older_group',
    'older_and_member',
    'new_inactive'
]

# 실험을 반복할 시드(seed) 목록입니다.
SEEDS = [42, 43, 44, 45, 46]

# 새로 생성 가능한 5개 피처의 전체 목록입니다. (수정하지 마세요)
ALL_NEW_FEATURES = [
    'is_older_group', 'older_and_member', 'is_low_frequency',
    'vip_inactive', 'new_inactive'
]


# ===================================================================

def run_and_log_experiment(dataset_path, features_to_include, seeds):
    """
    지정된 피처만 포함하여 실험을 수행하고 결과를 CSV 파일에 기록합니다.
    """
    # [수정] features_to_include 리스트가 비어 있는지 확인합니다.
    if not features_to_include:
        included_features_str = 'Original_Baseline'
    else:
        included_features_str = ', '.join(features_to_include)

    print("===== 피처 실험을 시작합니다 =====")
    print(f"사용 데이터셋: {dataset_path}")
    print(f"포함할 신규 피처: {included_features_str}")
    print(f"실행 시드: {seeds}")
    print("-" * 30)

    f1_scores, accuracy_scores = [], []

    full_df = pd.read_csv(dataset_path)

    features_to_exclude = [f for f in ALL_NEW_FEATURES if f not in features_to_include]
    existing_features_to_drop = [col for col in features_to_exclude if col in full_df.columns]
    df = full_df.drop(columns=existing_features_to_drop)

    temp_train_path = 'temp_train_for_experiment.csv'
    df.to_csv(temp_train_path, index=False)

    for seed in seeds:
        print(f"\n>>> 현재 시드: {seed}")
        result = catboost_classifier.train_and_eval(
            train_path=temp_train_path,
            seed=seed,
            produce_artifacts=False,
            params_dict={'verbose': 0}
        )

        f1_macro = result['metrics']['f1_macro']
        accuracy = result['metrics']['accuracy']
        f1_scores.append(f1_macro)
        accuracy_scores.append(accuracy)
        print(f"시드 {seed} -> F1-Macro: {f1_macro:.4f}, Accuracy: {accuracy:.4f}")

    os.remove(temp_train_path)

    mean_f1 = np.mean(f1_scores)
    mean_accuracy = np.mean(accuracy_scores)

    print("-" * 30)
    print(f"\n===== 최종 결과 =====")
    print(f"평균 F1-Macro CV 점수: {mean_f1:.4f}")
    print(f"평균 Accuracy CV 점수: {mean_accuracy:.4f}")

    log_results(included_features_str, mean_f1, mean_accuracy)


def log_results(included_features_str, f1_macro, accuracy):
    """
    실험 결과를 CSV 파일에 한 줄 추가합니다.
    """
    output_dir = 'results/eda/feature_engineering'
    log_file = os.path.join(output_dir, 'feature_experiments.csv')
    os.makedirs(output_dir, exist_ok=True)

    new_log = pd.DataFrame({
        'TIMESTAMP': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        '포함된 FEATURE': [included_features_str],
        'F1 MACRO': [f"{f1_macro:.4f}"],
        'ACCURACY': [f"{accuracy:.4f}"]
    })

    if not os.path.exists(log_file):
        new_log.to_csv(log_file, index=False, encoding='utf-8-sig')
        print(f"\n'{log_file}'을 생성하고 첫 결과를 기록했습니다.")
    else:
        new_log.to_csv(log_file, mode='a', header=False, index=False, encoding='utf-8-sig')
        print(f"\n'{log_file}'에 실험 결과를 추가로 기록했습니다.")


# --- 코드 실행 부분 ---
if __name__ == "__main__":
    run_and_log_experiment(
        dataset_path=BASE_FEATURED_DATASET,
        features_to_include=FEATURES_TO_INCLUDE,
        seeds=SEEDS
    )