import pandas as pd
import numpy as np
import os
from datetime import datetime
from itertools import combinations
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from tqdm import tqdm
import sys, os
from contextlib import contextmanager
from sklearn.metrics import f1_score


@contextmanager
def suppress_output():
    with open(os.devnull, 'w') as fnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = fnull, fnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

# ===================================================================
#                      사용자 설정 변수
# ===================================================================

# 탐색 방식: 'grid' 또는 'optuna'
SEARCHING_SWITCH = 'optuna'   # 'optuna' 로 바꿔서 실행 가능
USE_GPU = True
TRIALS = 2

# 원본 데이터 경로 (수정하지 않음)
BASE_FEATURED_DATASET = 'data/cattest_train.csv'

# 실험할 모델 이름 (수정하지 않음)
MODEL = 'catboost'  # 'catboost', 'xgboost' 등으로 변경하여 사용

BASE_FEATURE = [
    "ID", "age", "gender", "tenure", "frequent",
    "payment_interval", "subscription_type", "contract_length",
    "after_interaction", "support_needs"
]

full_df = pd.read_csv(BASE_FEATURED_DATASET)
EXPERIMENT_FEATURES = [col for col in full_df.columns if col not in BASE_FEATURE]

# 실험을 반복할 시드(seed) 목록
SEEDS = [45]
best_f1 = 0.0  # 수정 x

# ===================================================================

# [추가] MODEL 변수값에 따라 실제 사용할 모델 스크립트를 동적으로 import
if MODEL == 'lightgbm':
    import lightgbm_classifier as model_module
elif MODEL == 'catboost':
    import catboost_classifier as model_module
elif MODEL == 'xgboost':
    import xgboost_classifier as model_module
elif MODEL == 'tabnet':
    import tabnet_classifier as model_module
else:
    raise ValueError("지원하지 않는 모델입니다")


def run_and_log_experiment(model_module, dataset_path, features_to_include, seeds, log_file_path):
    """
    BASE_FEATURE + 선택된 features_to_include 조합으로 실험
    """
    final_features = BASE_FEATURE + features_to_include
    included_features_str = ', '.join(final_features)

    f1_scores, accuracy_scores = [], []
    full_df = pd.read_csv(dataset_path)


    # final_features만 남기고 나머지는 drop
    drop_cols = [c for c in full_df.columns if c not in final_features]
    df = full_df.drop(columns=drop_cols)

    temp_train_path = 'temp_train_for_experiment.csv'
    df.to_csv(temp_train_path, index=False)

    for seed in seeds:
        with suppress_output():
            result = model_module.train_and_eval(
                train_path=temp_train_path,
                seed=seed,
                produce_artifacts=False,
                use_gpu=USE_GPU
            )

        f1_scores.append(result['metrics']['f1_macro'])
        accuracy_scores.append(result['metrics']['accuracy'])


    os.remove(temp_train_path)

    mean_f1 = np.mean(f1_scores)
    mean_accuracy = np.mean(accuracy_scores)

    log_results(included_features_str, mean_f1, mean_accuracy, log_file_path)


def log_results(included_features_str, f1_macro, accuracy, log_file):
    new_log = pd.DataFrame({
        'TRIALS': [len(pd.read_csv(log_file)) + 1 if os.path.exists(log_file) else 1],
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

    if SEARCHING_SWITCH == 'grid':
        all_combinations = [[]]
        for r in range(1, len(EXPERIMENT_FEATURES) + 1):
            for combo in combinations(EXPERIMENT_FEATURES, r):
                all_combinations.append(list(combo))

        best_f1 = 0.0
        with tqdm(total=len(all_combinations), desc="Grid Search") as pbar:
            for features in all_combinations:
                run_and_log_experiment(
                    model_module=model_module,
                    dataset_path=BASE_FEATURED_DATASET,
                    features_to_include=features,
                    seeds=SEEDS,
                    log_file_path=log_file_path
                )
                last_f1 = float(pd.read_csv(log_file_path).iloc[-1]["F1 MACRO"])
                if last_f1 > best_f1:
                    best_f1 = last_f1
                pbar.set_postfix_str(f"BEST F1: {best_f1:.4f}")
                pbar.update(1)

    elif SEARCHING_SWITCH == 'optuna':
        best_f1 = 0.0


        def objective(trial):
            global best_f1
            selected = []

            # Trial 0 → baseline (BASE_FEATURE만)
            if trial.number == 0:
                full_df = pd.read_csv(BASE_FEATURED_DATASET)
                features = [col for col in full_df.columns if col not in BASE_FEATURE]
            else:
                for feat in EXPERIMENT_FEATURES:
                    use_feat = trial.suggest_int(f"use_{feat}", 0, 1)
                    if use_feat == 1:
                        selected.append(feat)
                features = selected

            run_and_log_experiment(
                model_module=model_module,
                dataset_path=BASE_FEATURED_DATASET,
                features_to_include=features,  # BASE_FEATURE는 함수 내부에서 자동 포함됨
                seeds=SEEDS,
                log_file_path=log_file_path
            )

            last_f1 = float(pd.read_csv(log_file_path).iloc[-1]["F1 MACRO"])
            if last_f1 > best_f1:
                best_f1 = last_f1

            return last_f1


        study = optuna.create_study(direction="maximize")
        with tqdm(total=TRIALS, desc="Optuna") as pbar:
            def wrapped_objective(trial):
                val = objective(trial)
                pbar.update(1)
                pbar.set_postfix_str(
                    f"trial {trial.number} best f1-macro : {best_f1:.4f} | current f1-macro : {val:.4f}")
                return val
            study.optimize(wrapped_objective, n_trials=TRIALS)

        print("Optuna 탐색 완료")

        results_df = pd.read_csv(log_file_path)
        best_idx = results_df["F1 MACRO"].astype(float).idxmax()
        best_trial = results_df.loc[best_idx, "TRIALS"]  # 몇 번째 trial인지
        best_features = results_df.loc[best_idx, "포함된 FEATURE"]

        if best_trial == 1:  # trial 0은 CSV에 TRIALS=1로 기록됨
            print("Best Features: BASE")
        else:
            print("Best Features:", best_features)

        print("Best score:", study.best_value)

    print("\n\n===== 모든 탐색 완료 =====")
    print(f"전체 결과는 '{log_file_path}'를 확인하세요.")