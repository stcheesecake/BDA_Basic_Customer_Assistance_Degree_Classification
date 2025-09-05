# lightgbm_hpo.py
# -*- coding: utf-8 -*-

"""
Optuna(TPE) for current multiclass lightgbm_classifier.py

요구사항:
- 실행 시 콘솔에는 tqdm 진행바만 표시 (추가 요약 print 전부 제거)
- lightgbm_classifier.py는 어떤 산출물도 생성하지 않음 (produce_artifacts=False)
- 결과는 단 하나의 CSV만 생성: results/optimization_lightgbm/YYMMDD_hhmmss_hpo.csv
- 목적함수: f1_macro 최대화
"""

TRIALS = 100

import os
import csv
import argparse
from datetime import datetime
import contextlib
import io
import json
import numpy as np
import optuna
from optuna.samplers import TPESampler
from tqdm import tqdm
optuna.logging.set_verbosity(optuna.logging.WARNING)

# [수정] 튜닝할 모델 스크립트 import
import lightgbm_classifier

# ───────────────────────── 검색 범위 (LightGBM 용으로 수정) ─────────────────────────
SEARCH_SPACE = dict(
    n_estimators=("int", 1500, 1700, 100),
    learning_rate=("float", 0.02, 0.05, None),  # None은 log-uniform 탐색
    num_leaves=("int", 145, 155, 1),
    max_depth=("int", 17, 18, 1),
    min_child_samples=("int", 99, 101, 1),
    subsample=("float", 0.8, 1.0, 0.2),
    colsample_bytree=("float", 0.9, 1.0, 0.1),
    reg_alpha=("float", 0.5, 1.5, None),  # L1 정규화 (log)
    reg_lambda=("float", 1e-3, 0.5, None),  # L2 정규화 (log)
)


# ─────────────────────────────────────────────────────────────────────
# Optuna Objective
# ─────────────────────────────────────────────────────────────────────
def _suggest_params(trial: optuna.Trial) -> dict:
    """SEARCH_SPACE에 정의된 범위에 따라 trial 파라미터를 제안합니다."""
    params = {}
    for name, (dtype, low, high, step) in SEARCH_SPACE.items():
        if dtype == "int":
            params[name] = trial.suggest_int(name, low, high, step=step)
        elif dtype == "float":
            if step is None:  # 로그 스케일
                params[name] = trial.suggest_float(name, low, high, log=True)
            else:
                params[name] = trial.suggest_float(name, low, high, step=step)
    return params


def objective(trial: optuna.Trial, args, csv_path):
    """
    Optuna의 각 trial에서 호출되는 목적 함수입니다.
    지정된 파라미터로 모델을 학습하고 f1_macro 점수를 반환합니다.
    """
    try:
        # 파라미터 제안
        params = _suggest_params(trial)

        # HPO 중에는 상세 로그를 출력하지 않도록 verbose=-1 추가
        params["verbose"] = -1

        # [수정] lightgbm_classifier.train_and_eval 호출
        # HPO 중에는 표준 출력을 모두 무시하여 tqdm 진행바만 보이게 함
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            out = lightgbm_classifier.train_and_eval(
                train_path=args.train_path,
                target_col="support_needs",
                save_dir=".",  # 의미 없음 (파일 저장 안 함)
                valid_size=args.valid_size,
                seed=args.seed,
                use_gpu=args.use_gpu,
                params_dict=params,  # dict 직접 전달
                produce_artifacts=False  # 🔴 어떤 파일도 생성하지 않음
            )

        metrics = out["metrics"]
        f1_macro = float(metrics.get("f1_macro", float("nan")))
        accuracy = float(metrics.get("accuracy", float("nan")))

        # 즉시 CSV에 기록
        row = [trial.number] + [params[k] for k in SEARCH_SPACE.keys()] + [f1_macro, accuracy]
        with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(row)

        return f1_macro

    except optuna.TrialPruned:
        raise
    except Exception as e:
        print(f"Trial #{trial.number} failed: {e}")
        return float("nan")  # 실패한 trial은 NaN 반환


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", default="data/train.csv")
    ap.add_argument("--n_trials", type=int, default=TRIALS)
    ap.add_argument("--valid_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_gpu", action="store_true")
    args = ap.parse_args()

    # 결과 저장 경로 설정
    timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
    save_dir = "results/lightgbm_optimization"  # [수정]
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f"{timestamp}_hpo.csv")

    # CSV 헤더 작성
    header = ["trial"] + list(SEARCH_SPACE.keys()) + ["f1_macro", "accuracy"]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(header)

    # Optuna Study 생성 및 최적화 실행
    sampler = TPESampler(seed=args.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    # tqdm을 사용한 진행바 표시
    with tqdm(total=args.n_trials, desc="Optimizing") as pbar:
        def callback(study, trial):
            pbar.update(1)

        study.optimize(
            lambda trial: objective(trial, args, csv_path),
            n_trials=args.n_trials,
            callbacks=[callback]
        )

    # 최종 결과 요약 출력
    print("\n\n===== HPO 완료 =====")
    print(f"총 Trial: {len(study.trials)}")
    print(f"최고 점수 (f1_macro): {study.best_value:.4f}")
    print("최적 파라미터:")
    for key, value in study.best_params.items():
        print(f"  - {key}: {value}")

    # 최고 파라미터를 json 파일로 저장
    best_params_path = os.path.join(save_dir, f"{timestamp}_best_params.json")
    with open(best_params_path, 'w') as f:
        json.dump(study.best_params, f, indent=4)
    print(f"\n최적 파라미터가 '{best_params_path}'에 저장되었습니다.")
    print(f"전체 결과는 '{csv_path}'를 확인하세요.")


if __name__ == "__main__":
    main()