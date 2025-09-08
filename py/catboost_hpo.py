# catboost_hpo.py
# -*- coding: utf-8 -*-

N_TRIALS = 2

import os
import csv
import json
import argparse
from datetime import datetime
import contextlib
import io
import catboost_classifier
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler

# ───────────────────────── 검색 범위 ─────────────────────────
SEARCH_SPACE = dict(
    iterations          = ("int",   1600, 1600, 100),
    learning_rate       = ("float", 0.13, 0.13, 0.01),
    depth               = ("int",   7,   7,    1),
    l2_leaf_reg         = ("float", 14.0, 14.0, 1.0),
    border_count        = ("int",   208,  208,  1),
    random_strength     = ("float", 0.0, 0.0,  0.1),
    bagging_temperature = ("float", 0.7, 0.7,  0.1),
)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", default="data/train.csv")
    ap.add_argument("--save_dir",   default="results/catboost_optimization")
    ap.add_argument("--trials",     type=int, default=N_TRIALS)
    ap.add_argument("--valid_size", type=float, default=0.2)
    ap.add_argument("--seed",       type=int, default=42)
    ap.add_argument("--use_gpu",    action="store_true")
    return ap.parse_args()

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def suggest(trial: optuna.Trial, name: str, spec):
    typ, lo, hi, step = spec
    if typ == "int":
        return trial.suggest_int(name, int(lo), int(hi), step=int(step))
    else:
        return trial.suggest_float(name, float(lo), float(hi), step=float(step))

def main():
    # Optuna 로그 최소화 (진행바만 보이게)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    args = parse_args()
    ensure_dir(args.save_dir)

    import catboost_classifier  # 현재 프로젝트용 catboost_classifier.py

    # 하나의 CSV만 생성
    tag = datetime.now().strftime("%y%m%d_%H%M%S")
    csv_path = os.path.join(args.save_dir, f"{tag}_hpo.csv")
    columns = ["trial", *[f"param_{k}" for k in SEARCH_SPACE.keys()], "f1_macro", "accuracy"]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerow(columns)

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=args.seed))

    def objective(trial: optuna.Trial):
        # 파라미터 샘플링
        params = {k: suggest(trial, k, spec) for k, spec in SEARCH_SPACE.items()}
        params.update({
            "loss_function": "MultiClass",
            "eval_metric": "TotalF1",
            "verbose": False,
            "task_type": "GPU" if args.use_gpu else "CPU",
            # HPO 단계: 제출/파일 생성 금지
            "submission": False,
        })

        # catboostclassifier.train_and_eval (무음 처리 + 파일 미생성)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            out = catboost_classifier.train_and_eval(
                train_path=args.train_path,
                target_col="support_needs",
                save_dir=".",                 # 의미 없음(파일 저장 안 함)
                valid_size=args.valid_size,
                seed=args.seed,
                use_gpu=args.use_gpu,
                params_dict=params,           # dict 직접 전달
                produce_artifacts=False       # 🔴 어떤 파일도 생성하지 않음
            )

        metrics = out["metrics"]
        f1_macro = float(metrics.get("f1_macro", float("nan")))
        accuracy = float(metrics.get("accuracy", float("nan")))

        # 즉시 CSV에 기록 (print 없음)
        row = [trial.number] + [params[k] for k in SEARCH_SPACE.keys()] + [f1_macro, accuracy]
        with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow(row)

        trial.set_user_attr("accuracy", accuracy)
        return f1_macro

    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    # ✅ 요약/경로 등 추가 print 전부 생략 (tqdm만 표시)

if __name__ == "__main__":
    main()
