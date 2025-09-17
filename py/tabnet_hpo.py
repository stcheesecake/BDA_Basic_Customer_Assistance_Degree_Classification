import os
import pandas as pd
import optuna
from tqdm import tqdm
from datetime import datetime
import argparse
import warnings  # [수정] warnings 모듈 import

# [수정] Optuna 정보 로그 숨김, UserWarning은 한 번만 출력
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('once', category=UserWarning)

from tabnet_classifier import train_and_eval as tabnet_train_and_eval

SAVE_DIR = "results/hpo"
os.makedirs(SAVE_DIR, exist_ok=True)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=2)
    parser.add_argument("--save_dir", type=str, default=SAVE_DIR)
    args = parser.parse_args()
    return args


def objective(trial):
    params = {
        "n_d": trial.suggest_int("n_d", 8, 32, step=4),
        "n_a": trial.suggest_int("n_a", 8, 32, step=4),
        "n_steps": trial.suggest_int("n_steps", 3, 10),
        "gamma": trial.suggest_float("gamma", 1.0, 2.0),
        "lambda_sparse": trial.suggest_float("lambda_sparse", 1e-6, 1e-3, log=True),
        "mask_type": trial.suggest_categorical("mask_type", ["sparsemax", "entmax"]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 5e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True),
        "virtual_batch_size": trial.suggest_categorical("virtual_batch_size", [128, 256, 512]),
        "train_path": "data/tabnet_train.csv",
        "test_path": "data/tabnet_test.csv",
        "seed": 45,
        "verbose": 0,
    }

    results = tabnet_train_and_eval(**params)
    f1_macro = results['metrics']['f1_macro']

    return f1_macro


if __name__ == '__main__':
    args = get_args()

    study = optuna.create_study(
        direction='maximize',
        study_name='tabnet_hpo'
    )

    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(args.save_dir, f"{now}_tabnet_hpo.csv")
    log_df = pd.DataFrame(columns=["number", "value", "params", "datetime_start"])

    for i in tqdm(range(args.n_trials), desc="Optimizing TabNet"):
        study.optimize(objective, n_trials=1)
        trial = study.trials[-1]

        log_df.loc[len(log_df)] = [
            trial.number,
            trial.value,
            trial.params,
            trial.datetime_start,
        ]
        log_df.to_csv(csv_path, index=False)

    best_params = study.best_params
    best_value = study.best_value

    print("\n\n==================================================")
    print("HPO가 완료되었습니다.")
    print(f"총 {args.n_trials}번의 trial 중 최적의 결과:")
    print(f"  - F1 Macro: {best_value:.4f}")
    print("  - 최적 하이퍼파라미터:")
    for key, value in best_params.items():
        print(f"    - {key}: {value}")

    print("\n최적의 하이퍼파라미터로 Submission 파일을 생성합니다...")
    best_params['submission'] = True
    best_params['verbose'] = 0  # [수정] 최종 학습 로그를 숨기기 위해 1 -> 0으로 변경

    final_results = tabnet_train_and_eval(**best_params)

    print("\n✅ 최종 모델 검증 점수:")
    print(f"  - F1 Macro: {final_results['metrics']['f1_macro']:.4f}")
    print(f"  - Accuracy: {final_results['metrics']['accuracy']:.4f}")
    print("==================================================")