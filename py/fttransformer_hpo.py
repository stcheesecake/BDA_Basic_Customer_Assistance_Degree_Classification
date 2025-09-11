# fttransformer_hpo.py
# -*- coding: utf-8 -*-

TRIALS = 1000

import os
import csv
import json
import argparse
from datetime import datetime
import contextlib
import io
import warnings
import logging
import numpy as np
import pandas as pd

import optuna
from optuna.samplers import TPESampler
from tqdm import tqdm

# 로그/경고 억제
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.transformer")

# FT-Transformer classifier (같은 폴더에 있어야 함)
import fttransformer_classifier


# ───────────────────────── Search Space ─────────────────────────
VALID_HEADS = {
    128: [2, 4, 8, 16],
    192: [2, 3, 6, 12],
    256: [2, 4, 8, 16],
    384: [2, 3, 6, 12],
    512: [2, 4, 8, 16],
}

SEARCH_SPACE_BASE = {
    # 모델 구조
    "d_model": ("categorical", [128, 192, 256, 384, 512]),
    "n_heads": ("categorical", None),  # d_model에 의존
    "n_layers": ("int", 2, 8, 1),
    "ff_mult": ("int", 4, 6, 1),

    # 드롭아웃
    "dropout": ("float", 0.0, 0.3, 0.05),
    "attn_dropout": ("float", 0.0, 0.3, 0.05),
    "token_dropout": ("float", 0.0, 0.3, 0.05),

    # 학습률/스케줄러
    "lr": ("float", 5e-4, 5e-3, None),            # log scale
    "weight_decay": ("float", 1e-6, 1e-3, None),  # log scale
    "warmup_ratio": ("float", 0.0, 0.2, 0.01),
    "min_lr": ("float", 1e-5, 1e-4, None),        # log scale
    "step_size": ("int", 5, 20, 5),
    "gamma": ("float", 0.5, 0.9, 0.1),

    # 손실
    "label_smoothing": ("float", 0.0, 0.1, 0.01),
    "focal_gamma": ("float", 1.0, 3.0, 0.5),

    # 배치/에폭
    "batch_size": ("int", 128, 256, 64),
    "epochs": ("int", 40, 100, 5),

    # 유틸
    "grad_clip": ("float", 0.5, 2.0, 0.5),
    "patience": ("int", 10, 30, 5),
}

ALPHA_DIM = None  # main()에서 데이터로 파악
ALPHA_RANGE = (0.5, 1.5, 0.05)  # (low, high, step)


def _suggest_params(trial: optuna.Trial, search_space: dict) -> dict:
    params = {}
    # d_model 먼저
    stype, choices = search_space["d_model"]
    params["d_model"] = trial.suggest_categorical("d_model", choices)

    # d_model에 맞는 n_heads
    params["n_heads"] = trial.suggest_categorical("n_heads", VALID_HEADS[params["d_model"]])

    for name, spec in search_space.items():
        if name in ("d_model", "n_heads"):
            continue
        stype = spec[0]
        if stype == "categorical":
            _, c = spec
            params[name] = trial.suggest_categorical(name, c)
        elif stype == "int":
            _, low, high, step = spec
            params[name] = trial.suggest_int(name, low, high, step=step)
        elif stype == "float":
            _, low, high, step = spec
            if step is None:
                params[name] = trial.suggest_float(name, low, high, log=True)
            else:
                params[name] = trial.suggest_float(name, low, high, step=step)

    # 안전 제약
    if params["min_lr"] >= params["lr"]:
        params["min_lr"] = max(1e-6, params["lr"] * 0.1)

    if params["dropout"] + params["attn_dropout"] + params["token_dropout"] > 0.7:
        trio = ["dropout", "attn_dropout", "token_dropout"]
        biggest = max(trio, key=lambda k: params[k])
        params[biggest] = max(0.0, params[biggest] - 0.2)

    if params["patience"] > params["epochs"]:
        params["patience"] = max(1, params["epochs"] // 2)

    return params


def _maybe_make_alpha_vec(params: dict, alpha_dim: int):
    keys = [f"alpha{i}" for i in range(alpha_dim)]
    if all(k in params for k in keys):
        vec = [float(params[k]) for k in keys]
        vec = np.clip(vec, 1e-8, None).tolist()
        return ",".join(f"{v:.4f}" for v in vec)
    return ""


def _extract_class_metrics(metrics: dict, k: int):
    """metrics에서 per-class f1/precision/recall과 macro precision/recall을 안전하게 추출."""
    # macro
    f1_macro = float(metrics.get("f1_macro", 0.0) or 0.0)
    prec_macro = float(metrics.get("precision_macro", metrics.get("precision", 0.0)) or 0.0)
    rec_macro = float(metrics.get("recall_macro", metrics.get("recall", 0.0)) or 0.0)
    acc = float(metrics.get("accuracy", 0.0) or 0.0)

    # per-class 컨테이너 탐색
    per = metrics.get("per_class") or metrics.get("by_class") or {}
    f1_arr = per.get("f1") or metrics.get("f1_per_class") or metrics.get("class_f1") or []
    p_arr = per.get("precision") or metrics.get("precision_per_class") or metrics.get("class_precision") or []
    r_arr = per.get("recall") or metrics.get("recall_per_class") or metrics.get("class_recall") or []

    # 길이 보정
    def _fix(arr):
        if not isinstance(arr, (list, tuple, np.ndarray)):
            return [0.0] * k
        arr = list(arr)
        if len(arr) < k:
            arr = arr + [0.0] * (k - len(arr))
        elif len(arr) > k:
            arr = arr[:k]
        return [float(x) for x in arr]

    f1_arr = _fix(f1_arr)
    p_arr = _fix(p_arr)
    r_arr = _fix(r_arr)

    return f1_macro, prec_macro, rec_macro, acc, f1_arr, p_arr, r_arr


def objective(trial: optuna.Trial, cli_args, csv_path, search_space: dict, alpha_dim: int):
    logging.disable(logging.CRITICAL)
    try:
        params = _suggest_params(trial, search_space)

        base_args = fttransformer_classifier.parse_args()
        alpha_vec_str = _maybe_make_alpha_vec(params, alpha_dim) if alpha_dim else ""

        trial_overrides = {k: v for k, v in params.items() if not k.startswith("alpha")}
        trial_overrides["use_focal"] = True
        trial_overrides["alpha_vec"] = alpha_vec_str

        merged = {**vars(base_args), **vars(cli_args), **trial_overrides}
        if "valid_size" in merged:
            merged["val_ratio"] = merged["valid_size"]
        hpo_args = argparse.Namespace(**merged)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            out = fttransformer_classifier.train_and_eval(
                hpo_args, device="cuda" if cli_args.use_gpu else "cpu"
            )

        if not out or "metrics" not in out:
            trial.set_user_attr("metrics", {})
            return 0.0

        metrics = out["metrics"]

        # 클래스 수(K)
        K = alpha_dim if alpha_dim else int(metrics.get("num_classes", 0) or 0)
        # 안전: K가 0이면 추정 불가 → per-class는 빈값
        f1_macro, prec_macro, rec_macro, acc, f1_arr, p_arr, r_arr = _extract_class_metrics(metrics, K if K > 0 else 0)

        # CSV 쓰기
        ordered_keys = [k for k in search_space.keys()]
        if alpha_dim:
            ordered_keys += [f"alpha{i}" for i in range(alpha_dim)]

        row = [trial.number] + [params.get(k, "") for k in ordered_keys]
        # 사람이 읽기 좋은 alpha_vec
        if alpha_dim:
            row += [alpha_vec_str]

        # 메트릭: f1_macro → per-class f1 → per-class precision → per-class recall → precision_macro, recall_macro → accuracy
        row += [f1_macro]
        if K and (len(f1_arr) == K and len(p_arr) == K and len(r_arr) == K):
            row += [*f1_arr, *p_arr, *r_arr, prec_macro, rec_macro, acc]
        else:
            row += []  # 헤더 길이 맞추기는 main()에서 헤더 구성으로 해결

        with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow(row)

        # best trial 요약 출력을 위해 user_attrs에 저장
        trial.set_user_attr("metrics", {
            "f1_macro": f1_macro,
            "precision_macro": prec_macro,
            "recall_macro": rec_macro,
            "accuracy": acc,
            "f1_per_class": f1_arr,
            "precision_per_class": p_arr,
            "recall_per_class": r_arr,
        })

        return f1_macro

    except Exception:
        trial.set_user_attr("metrics", {})
        return 0.0
    finally:
        logging.disable(logging.NOTSET)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", default="data/train.csv")
    ap.add_argument("--n_trials", type=int, default=TRIALS)
    ap.add_argument("--valid_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_gpu", action="store_true", default=True)
    ap.add_argument("--test_path", default="data/test.csv")
    ap.add_argument("--target", default="support_needs")

    # alpha_vec 튜닝
    ap.add_argument("--tune_alpha_vec", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--alpha_low", type=float, default=ALPHA_RANGE[0])
    ap.add_argument("--alpha_high", type=float, default=ALPHA_RANGE[1])
    ap.add_argument("--alpha_step", type=float, default=ALPHA_RANGE[2])

    args = ap.parse_args()

    # 저장 경로
    timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
    save_dir = "results/optimization_fttransformer"
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f"{timestamp}_hpo.csv")
    best_params_path = os.path.join(save_dir, f"{timestamp}_best_params.json")

    # 클래스 수 파악
    alpha_dim = None
    try:
        df_tmp = pd.read_csv(args.train_path)
        if args.target not in df_tmp.columns:
            raise ValueError(f"target '{args.target}' not in train file.")
        _, inv = np.unique(df_tmp[args.target].to_numpy(), return_inverse=True)
        c = len(np.unique(inv))
        alpha_dim = c if c >= 2 else None
    except Exception:
        alpha_dim = None

    # search space 구성
    search_space = dict(SEARCH_SPACE_BASE)
    if args.tune_alpha_vec and alpha_dim:
        lo, hi, st = args.alpha_low, args.alpha_high, args.alpha_step
        for i in range(alpha_dim):
            search_space[f"alpha{i}"] = ("float", lo, hi, st)

    # CSV 헤더 구성
    header = ["trial"] + list(search_space.keys())
    if args.tune_alpha_vec and alpha_dim:
        header += ["alpha_vec"]
    # 메트릭 헤더: f1_macro → f1_c* → precision_c* → recall_c* → precision_macro, recall_macro, accuracy
    header += ["f1_macro"]
    if alpha_dim:
        header += [f"f1_c{i}" for i in range(alpha_dim)]
        header += [f"precision_c{i}" for i in range(alpha_dim)]
        header += [f"recall_c{i}" for i in range(alpha_dim)]
        header += ["precision_macro", "recall_macro", "accuracy"]
    else:
        # alpha_dim을 모르면 최소한 f1_macro만 기록(정상 추정되면 위 컬럼이 붙음)
        pass

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerow(header)

    # Optuna
    sampler = TPESampler(seed=args.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    with tqdm(total=args.n_trials, desc="Optimizing", leave=True) as pbar:
        def _cb(study_obj: optuna.Study, trial_obj: optuna.trial.FrozenTrial):
            best = 0.0
            if study_obj.best_trial is not None and study_obj.best_trial.value is not None:
                best = study_obj.best_trial.value
            pbar.set_postfix_str(f"best_f1_macro : {best:.4f}")
            pbar.update(1)

        study.optimize(lambda t: objective(t, args, csv_path, search_space, alpha_dim),
                       n_trials=args.n_trials,
                       callbacks=[_cb])

    # 완료 요약
    print("\n===== HPO 완료 =====")
    if not study.trials or study.best_trial is None or study.best_value is None:
        print("성공적으로 완료된 Trial이 없습니다.")
    else:
        best_trial = study.best_trial
        best = dict(best_trial.params)
        print(f"총 Trial: {len(study.trials)}")
        # 메트릭 요약 (요청 포맷)
        m = best_trial.user_attrs.get("metrics", {}) or {}
        f1_macro = float(m.get("f1_macro", 0.0) or 0.0)
        print(f"최고 점수 (f1_macro): {f1_macro:.4f}")

        f1_pc = m.get("f1_per_class") or []
        p_pc = m.get("precision_per_class") or []
        r_pc = m.get("recall_per_class") or []
        if f1_pc and p_pc and r_pc:
            print("클래스별:")
            for i in range(len(f1_pc)):
                print(f"  - class {i}: f1 {f1_pc[i]:.4f}, precision {p_pc[i]:.4f}, recall {r_pc[i]:.4f}")

        prec_macro = float(m.get("precision_macro", 0.0) or 0.0)
        rec_macro = float(m.get("recall_macro", 0.0) or 0.0)
        print(f"precision_macro : {prec_macro:.4f}")
        print(f"recall_macro : {rec_macro:.4f}")

        # 보기 좋게 alpha_vec 문자열도 출력
        alpha_dim_out = sum(1 for k in best.keys() if k.startswith("alpha"))
        if alpha_dim_out:
            alpha_vec = [best.pop(f"alpha{i}") for i in range(alpha_dim_out)]
            print(f"alpha_vec: [{', '.join(f'{v:.4f}' for v in alpha_vec)}]")

        print("최적 파라미터:")
        for k, v in best.items():
            print(f"  - {k}: {v}")

        with open(best_params_path, "w", encoding="utf-8") as f:
            json.dump(best_trial.params, f, indent=4, ensure_ascii=False)
        print(f"\n최적 파라미터가 '{best_params_path}'에 저장되었습니다.")
    print(f"전체 결과는 '{csv_path}'를 확인하세요.")


if __name__ == "__main__":
    main()
