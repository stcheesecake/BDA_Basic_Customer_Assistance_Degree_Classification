import os
import pandas as pd
import optuna
from tqdm import tqdm
from datetime import datetime
import argparse
import warnings
from contextlib import redirect_stdout

# ==================================================
# 최종 Submission 파일 생성 여부를 결정하는 스위치
SUBMISSION_SWITCH = False
# ==================================================

optuna.logging.set_verbosity(optuna.logging.ERROR)
warnings.filterwarnings('ignore', category=UserWarning)

from tabnet_classifier import train_and_eval as tabnet_train_and_eval

SAVE_DIR = "results/optimization_tabnet"
os.makedirs(SAVE_DIR, exist_ok=True)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=1000)
    parser.add_argument("--save_dir", type=str, default=SAVE_DIR)
    args = parser.parse_args()
    return args

def objective(trial):
    """Optuna가 최적화할 목적 함수"""

    # [수정] n_d와 n_a를 n_d_a로 통합하여 같은 값을 사용하도록 변경
    n_d_a = trial.suggest_int("n_d_a", 4, 64, step=4)

    params = {
        # [통합] n_d와 n_a를 하나의 파라미터로 통합
        "n_d": n_d_a,
        "n_a": n_d_a,
        "n_steps": trial.suggest_int("n_steps", 3, 10),
        "gamma": trial.suggest_float("gamma", 1.0, 3.0),
        "lambda_sparse": trial.suggest_float("lambda_sparse", 1e-6, 1e-3, log=True),
        "mask_type": trial.suggest_categorical("mask_type", ["sparsemax", "entmax"]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 5e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True),
        "virtual_batch_size": trial.suggest_categorical("virtual_batch_size", [128, 256, 512]),
        "cat_emb_dim": trial.suggest_int("cat_emb_dim", 1, 4),
        "n_independent": trial.suggest_int("n_independent", 1, 5),
        "n_shared": trial.suggest_int("n_shared", 1, 5),
        "patience": trial.suggest_int("patience", 20, 70),
        "batch_size": trial.suggest_categorical("batch_size", [512, 1024, 2048]),

        # 고정 파라미터 (HPO 대상이 아님)
        "train_path": "data/tabnet_train.csv",
        "test_path": "data/tabnet_test.csv",
        "seed": 45,
        "verbose": 0,
        "max_epochs": 200,  # [수정] max_epochs 값을 명시적으로 전달
    }

    # ⭐️ [수정] 학습 함수 호출 부분을 redirect_stdout으로 감싸서 출력 제어
    with redirect_stdout(open(os.devnull, 'w')):
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

    pbar = tqdm(range(args.n_trials), desc="Optimizing TabNet")
    for i in pbar:
        study.optimize(objective, n_trials=1)
        trial = study.trials[-1]

        log_df.loc[len(log_df)] = [
            trial.number,
            trial.value,
            trial.params,
            trial.datetime_start,
        ]
        log_df.to_csv(csv_path, index=False)
        pbar.set_postfix({"best_f1": f"{study.best_value:.4f}"})

    best_params = study.best_params
    best_value = study.best_value

    print("\n\n==================================================")
    print("HPO가 완료되었습니다.")
    print(f"총 {args.n_trials}번의 trial 중 최적의 결과:")
    print(f"  - F1 Macro: {best_value:.4f}")
    print("  - 최적 하이퍼파라미터:")
    for key, value in best_params.items():
        print(f"    - {key}: {value}")
    print("==================================================")

    if SUBMISSION_SWITCH:
        print("\n최적의 하이퍼파라미터로 Submission 파일을 생성합니다...")

        # Optuna는 n_d_a만 저장하므로, 실제 train 함수에 맞게 n_d, n_a를 다시 만들어줘야 함
        final_params = best_params.copy()
        n_d_a_val = final_params.pop('n_d_a') # n_d_a는 제거
        final_params['n_d'] = n_d_a_val
        final_params['n_a'] = n_d_a_val

        final_params['submission'] = True
        final_params['verbose'] = 0

        final_results = tabnet_train_and_eval(**final_params)

        print("\n✅ 최종 모델 검증 점수:")
        print(f"  - F1 Macro: {final_results['metrics']['f1_macro']:.4f}")
        print(f"  - Accuracy: {final_results['metrics']['accuracy']:.4f}")
        print("==================================================")
    else:
        print("\nSubmission 파일 생성을 건너뛰었습니다.")
        print("==================================================")