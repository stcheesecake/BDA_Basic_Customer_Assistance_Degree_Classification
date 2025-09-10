# lightgbm_hpo.py
# -*- coding: utf-8 -*-


N_TRIALS = 1000

import os
import csv
import json
import argparse
from datetime import datetime
import contextlib
import io

import optuna
from optuna.samplers import TPESampler
from tqdm import tqdm

optuna.logging.set_verbosity(optuna.logging.WARNING)

# [수정] 튜닝할 모델 스크립트 import
import lightgbm_classifier

# ───────────────────────── 검색 범위 (기존과 동일) ─────────────────────────
SEARCH_SPACE = dict(
    n_estimators=("int", 500, 2000, 100),
    learning_rate=("float", 0.01, 0.1, None),
    num_leaves=("int", 20, 150, 10),
    max_depth=("int", 5, 12, 1),
    min_child_samples=("int", 10, 100, 10),
    subsample=("float", 0.6, 1.0, 0.1),
    colsample_bytree=("float", 0.6, 1.0, 0.1),
    reg_alpha=("float", 1e-3, 10.0, None),
    reg_lambda=("float", 1e-3, 10.0, None),
)


# ─────────────────────────────────────────────────────────────────────
# Optuna Objective (단일 분할 방식으로 수정)
# ─────────────────────────────────────────────────────────────────────
def _suggest_params(trial: optuna.Trial) -> dict:
    params = {}
    for name, (dtype, low, high, step) in SEARCH_SPACE.items():
        if dtype == "int":
            params[name] = trial.suggest_int(name, low, high, step=step)
        elif dtype == "float":
            if step is None:
                params[name] = trial.suggest_float(name, low, high, log=True)
            else:
                params[name] = trial.suggest_float(name, low, high, step=step)
    return params


def objective(trial: optuna.Trial, args, csv_path):
    """
    [수정] lightgbm_classifier를 직접 호출하여 단일 분할 점수를 사용합니다.
    """
    try:
        params = _suggest_params(trial)
        params["verbose"] = -1

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            out = lightgbm_classifier.train_and_eval(
                train_path=args.train_path,
                target_col="support_needs",
                save_dir=".",
                valid_size=args.valid_size,
                seed=args.seed,
                use_gpu=args.use_gpu,
                params_dict=params,
                produce_artifacts=False
            )

        metrics = out["metrics"]
        f1_macro = float(metrics.get("f1_macro", float("nan")))
        accuracy = float(metrics.get("accuracy", float("nan")))

        row = [trial.number] + [params[k] for k in SEARCH_SPACE.keys()] + [f1_macro, accuracy]
        with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(row)

        return f1_macro

    except optuna.TrialPruned:
        raise
    except Exception as e:
        return float("nan")


# ─────────────────────────────────────────────────────────────────────
# CLI (기존과 동일)
# ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", default="data/train.csv")
    ap.add_argument("--n_trials", type=int, default=N_TRIALS)
    ap.add_argument("--valid_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_gpu", action="store_true")
    args = ap.parse_args()

    timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
    save_dir = "results/optimization_lightgbm"
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f"{timestamp}_hpo.csv")

    header = ["trial"] + list(SEARCH_SPACE.keys()) + ["f1_macro", "accuracy"]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(header)

    sampler = TPESampler(seed=args.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    with tqdm(total=args.n_trials, desc="Optimizing") as pbar:
        def callback(study, trial):
            pbar.update(1)

        study.optimize(
            lambda trial: objective(trial, args, csv_path),
            n_trials=args.n_trials,
            callbacks=[callback]
        )

    print("\n\n===== HPO 완료 =====")
    if not study.trials or study.best_trial is None:
        print("성공적으로 완료된 Trial이 없습니다.")
    else:
        print(f"총 Trial: {len(study.trials)}")
        print(f"최고 점수 (f1_macro): {study.best_value:.4f}")
        print("최적 파라미터:")
        for key, value in study.best_params.items():
            print(f"  - {key}: {value}")

        best_params_path = os.path.join(save_dir, f"{timestamp}_best_params.json")
        with open(best_params_path, 'w') as f:
            json.dump(study.best_params, f, indent=4)
        print(f"\n최적 파라미터가 '{best_params_path}'에 저장되었습니다.")

    print(f"전체 결과는 '{csv_path}'를 확인하세요.")


if __name__ == "__main__":
    main()