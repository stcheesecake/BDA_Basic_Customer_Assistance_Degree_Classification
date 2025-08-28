# hpo.py
# -*- coding: utf-8 -*-

"""
Bayesian Optimization (TPE) for CatBoost via model.train_and_eval()

- 탐색 기준: balacc 최대화
- 기록 기준: F1 최고(best_trial.txt), balacc 최고(best_params.json)
- 화면 출력: tqdm 진행률바 1줄 + postfix 1줄(자동 줄바꿈은 콘솔에 맡김)
- model.py는 produce_artifacts=False, quiet=True 로 호출
- 각 trial 결과를 즉시 CSV에 append (중도 종료 대비)
"""

import os
import io
import json
import csv
import argparse
from datetime import datetime
import contextlib

import pandas as pd
from tqdm import tqdm
import optuna
from optuna.samplers import TPESampler


# ────────────────────────────────────────────────────────────────────
# (🔧) 최상단에서 하이퍼파라미터 범위와 trial 수를 지정
# 각 항목은 [start, end, step] 형식
PARAMS_CONFIG = {
    "iterations":          [1000, 3000, 200],  # int
    "learning_rate":       [0.1, 0.3, 0.01],   # float
    "depth":               [6, 9, 1],          # int
    "l2_leaf_reg":         [1.0, 20.0, 0.5],   # float
    "border_count":        [50, 200, 10],      # int
    "random_strength":     [1.0, 5.0, 0.1],    # float
    "bagging_temperature": [0.1, 0.9, 0.05],   # float

    # ⬇️ 추가: SMOTE-NC 하이퍼파라미터도 탐색
    "smote_sampling":      [0.80, 1.00, 0.05], # float (소수:다수 목표 비율)
    "smote_k":             [3, 9, 1],          # int   (k_neighbors)
}
TRIALS = 2000
SEP = " | "  # 한 줄 출력용 구분자
# ────────────────────────────────────────────────────────────────────


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", default="data/preprocessed_train_oof.csv")
    ap.add_argument("--save_dir", default="results/optimization")
    ap.add_argument("--csv_path", default="results/optimization/bo_trials.csv")
    ap.add_argument("--params_json_out", default="results/optimization/best_params.json")
    ap.add_argument("--best_txt_out", default="results/optimization/best_trial.txt")
    ap.add_argument("--valid_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--trials", type=int, default=TRIALS)
    return ap.parse_args()


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _suggest_from_config(trial: optuna.Trial, name: str, cfg):
    s, e, st = cfg
    if float(st).is_integer() and float(s).is_integer() and float(e).is_integer():
        return trial.suggest_int(name, int(s), int(e), step=int(st))
    return trial.suggest_float(name, float(s), float(e), step=float(st))


def build_params_from_trial(trial: optuna.Trial) -> dict:
    params = {k: _suggest_from_config(trial, k, cfg) for k, cfg in PARAMS_CONFIG.items()}
    params.update({
        "task_type": "GPU",
        "boosting_type": "Ordered",
        "bootstrap_type": "Bayesian",
        "verbose": False,
    })
    return params


def main():
    # Optuna INFO 로그 숨김
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    args = parse_args()
    _ensure_dir(args.save_dir)

    # 같은 폴더의 model.py 사용
    import model

    # ── (NEW) 실행마다 전용 폴더/파일 경로 만들기 ─────────────────────────
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(args.save_dir, run_tag)
    _ensure_dir(results_dir)

    # 입력 인자에 있던 경로들은 그대로 두되, 실제 쓰기는 run 전용 파일로
    run_csv_path       = os.path.join(results_dir, f"{run_tag}_bo_trials.csv")
    run_params_json    = os.path.join(results_dir, f"{run_tag}_best_params.json")
    run_best_txt       = os.path.join(results_dir, f"{run_tag}_best_trial.txt")

    # CSV 헤더 생성 (실행마다 새로운 파일)
    columns = [
        "trial", *[f"param_{k}" for k in PARAMS_CONFIG.keys()],
        "best_threshold", "policy",
        "f1", "auc", "precision", "recall",
        "acc0", "acc1", "balacc", "youden",
        "score", "seed",
    ]
    with open(run_csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter=",", quoting=csv.QUOTE_NONNUMERIC, lineterminator="\n")
        writer.writerow(columns)

    # 베스트 트래커
    best_balacc = -1.0
    best_by_balacc = None  # (trial_no, params, metrics, thr)
    best_f1 = -1.0
    best_by_f1 = None      # (trial_no, params, metrics, thr)

    # ── 진행률바 1줄만 사용 (막대 길이 고정 4칸; 나머지는 콘솔이 자동 줄바꿈)
    pbar = tqdm(
        total=args.trials,
        desc="Bayesian Optimization (TPE)",
        dynamic_ncols=True,
        bar_format=(
            "{desc:<26} "
            "{percentage:3.0f}%|{bar:4}| "
            "{n_fmt}/{total_fmt} [{elapsed}<{remaining}]  {postfix}"
        ),
        position=0,
        leave=True,
    )

    def objective(trial: optuna.Trial):
        nonlocal best_balacc, best_by_balacc, best_f1, best_by_f1

        params = build_params_from_trial(trial)

        # (NEW) SMOTE-NC 메타키 매핑: model.train_and_eval이 인식
        #  - PARAMS_CONFIG의 smote_* 값을 pop해 CatBoost로 전달되지 않게 함
        params["_use_smote_nc"]   = True
        params["_smote_sampling"] = float(params.pop("smote_sampling"))
        params["_smote_k"]        = int(params.pop("smote_k"))

        # model.py 출력/저장 차단
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            out = model.train_and_eval(
                train_path=args.train_path,
                params=params,
                target_col="withdrawal",
                save_dir=results_dir,          # (NEW) 이 실행 전용 폴더
                valid_size=args.valid_size,
                seed=args.seed,
                deterministic=args.deterministic,
                produce_artifacts=False,       # 파일 생성 X
                quiet=True,                    # 콘솔 출력 X
            )

        thr = float(out["threshold"])
        m = out["metrics"]
        f1 = float(m.get("f1", float("nan")))
        balacc = float(m.get("balacc", float("nan")))
        policy = getattr(model, "THRESHOLD_STRATEGY", "balacc")

        # CSV 즉시 append 저장 (quoted)
        param_values = {f"param_{k}": trial.params.get(k, None) for k in PARAMS_CONFIG.keys()}

        row = {
            "trial": trial.number,
            **param_values,
            "best_threshold": thr, "policy": policy,
            "f1": m.get("f1"), "auc": m.get("auc"),
            "precision": m.get("precision"), "recall": m.get("recall"),
            "acc0": m.get("acc0"), "acc1": m.get("acc1"),
            "balacc": m.get("balacc"), "youden": m.get("youden"),
            "score": m.get("score"),
            "seed": args.seed,
        }
        with open(run_csv_path, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f, delimiter=",", quoting=csv.QUOTE_NONNUMERIC, lineterminator="\n")
            writer.writerow([row[c] for c in columns])

        # 베스트 갱신
        if balacc > best_balacc:
            best_balacc = balacc
            best_by_balacc = (trial.number, {**trial.params}, m, thr)
        if f1 > best_f1:
            best_f1 = f1
            best_by_f1 = (trial.number, {**trial.params}, m, thr)

        # ── 출력: 진행률바 한 줄의 postfix 로만 갱신
        postfix = SEP.join([
            f"thr:{thr:.4f}",
            f"f1:{float(m.get('f1', float('nan'))):.4f}",
            f"bal:{float(m.get('balacc', float('nan'))):.4f}",
            f"auc:{float(m.get('auc', float('nan'))):.4f}",
            f"pre:{float(m.get('precision', float('nan'))):.4f}",
            f"rec:{float(m.get('recall', float('nan'))):.4f}",
            f"a0:{float(m.get('acc0', float('nan'))):.4f}",
            f"a1:{float(m.get('acc1', float('nan'))):.4f}",
            f"you:{float(m.get('youden', float('nan'))):.4f}",
            f"sc:{float(m.get('score', float('nan'))):.4f}",
            f"sd:{args.seed}",
        ])
        pbar.set_postfix_str(postfix, refresh=True)
        pbar.update(1)

        # 최적화 기준: balacc
        return balacc

    # Optuna study
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=args.seed))

    try:
        study.optimize(objective, n_trials=args.trials, show_progress_bar=False, gc_after_trial=True)
    except KeyboardInterrupt:
        print("\n[Warn] Interrupted by user. Writing artifacts collected so far...")
    finally:
        pbar.close()

        # 저장물
        if best_by_balacc is not None:
            tno, params_used, metrics, thr = best_by_balacc
            payload = {
                "params": {
                    **params_used,
                    "_use_smote_nc": True,
                    "_smote_sampling": float(params_used.get("smote_sampling")),
                    "_smote_k": int(params_used.get("smote_k")),
                },
                "best": {
                    "trial": int(tno),
                    "threshold": float(thr),
                    "policy": getattr(model, "THRESHOLD_STRATEGY", "balacc"),
                    "metrics": metrics,
                },
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "selection_metric": "balacc",
            }
            with open(run_params_json, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            print(f"\n[Saved] best params (by balacc) -> {run_params_json}")

        if best_by_f1 is not None:
            tno, params_used, metrics, thr = best_by_f1
            lines = [
                "==== Best Trial (by F1) ====",
                f"trial: {tno}",
                f"Best threshold : {thr:.4f} (policy: {getattr(model, 'THRESHOLD_STRATEGY', 'balacc')})",
                "",
                "[Metrics]",
            ]
            for k in ["f1", "auc", "precision", "recall", "acc0", "acc1", "balacc", "youden", "score"]:
                v = metrics.get(k, None)
                if v is not None:
                    try:
                        lines.append(f"{k}: {float(v):.6f}")
                    except Exception:
                        lines.append(f"{k}: {v}")
            lines.append("")
            lines.append("[Params]")
            # trial.params에는 smote_*가 들어있음
            for k in sorted(params_used.keys()):
                lines.append(f"{k}: {params_used[k]}")
            with open(run_best_txt, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            print(f"[Saved] best trial (by F1) -> {run_best_txt}")


if __name__ == "__main__":
    main()
