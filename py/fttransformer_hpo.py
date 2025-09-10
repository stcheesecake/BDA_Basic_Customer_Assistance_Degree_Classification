# fttransformer_hpo.py
# -*- coding: utf-8 -*-

TRIALS = 3

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

import optuna
from optuna.samplers import TPESampler
from tqdm import tqdm

# 외부 경고/로그 억제
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.transformer")

# FT-Transformer classifier 모듈 (동일 디렉토리에 있어야 합니다)
import fttransformer_classifier


# ───────────────────────── Search Space ─────────────────────────
# d_model별 허용 가능한 n_heads (d_model % n_heads == 0 보장)
VALID_HEADS = {
    128: [2, 4, 8, 16],
    192: [2, 3, 6, 12],
    256: [2, 4, 8, 16],
    384: [2, 3, 6, 12],
    512: [2, 4, 8, 16],
}

# 형식:
#  - ("categorical", [choices])
#  - ("int", low, high, step)
#  - ("float", low, high, step or None[=log scale])
SEARCH_SPACE = {
    # 모델 구조
    "d_model": ("categorical", [128, 192, 256, 384, 512]),
    "n_heads": ("categorical", None),  # d_model에 따라 VALID_HEADS에서 선택
    "n_layers": ("int", 2, 8, 1),
    "ff_mult": ("int", 2, 8, 1),

    # 드롭아웃
    "dropout": ("float", 0.0, 0.3, 0.05),
    "attn_dropout": ("float", 0.0, 0.3, 0.05),
    "token_dropout": ("float", 0.0, 0.3, 0.05),

    # 학습률/스케줄러
    "lr": ("float", 1e-4, 1e-3, None),           # log scale
    "weight_decay": ("float", 1e-6, 1e-3, None), # log scale
    "warmup_ratio": ("float", 0.0, 0.2, 0.05),
    "min_lr": ("float", 1e-6, 1e-5, None),       # log scale (lr보다 항상 작게 보정)
    "step_size": ("int", 5, 20, 5),
    "gamma": ("float", 0.5, 0.9, 0.1),

    # 손실
    "label_smoothing": ("float", 0.0, 0.1, 0.05),
    "focal_gamma": ("float", 1.0, 3.0, 0.5),
    "focal_alpha": ("float", 0.2, 0.8, 0.2),

    # 학습 배치/에폭
    "batch_size": ("int", 128, 512, 64),
    "epochs": ("int", 1, 2, 1),  # 의미 있는 학습을 위해 최소 10

    # 유틸
    "grad_clip": ("float", 0.5, 5.0, 0.5),
    "patience": ("int", 5, 20, 5),
}


def _suggest_params(trial: optuna.Trial) -> dict:
    """SEARCH_SPACE를 동적으로 해석하여 optuna 파라미터를 제안합니다.
       d_model을 먼저 고른 뒤, 그에 맞는 n_heads를 VALID_HEADS에서 고릅니다.
    """
    params = {}

    # 1) d_model 먼저 선택
    stype, choices = SEARCH_SPACE["d_model"]
    assert stype == "categorical"
    params["d_model"] = trial.suggest_categorical("d_model", choices)

    # 2) n_heads는 d_model에 따라 유효 후보에서 선택 (invalid 조합 생성 자체 차단)
    params["n_heads"] = trial.suggest_categorical("n_heads", VALID_HEADS[params["d_model"]])

    # 3) 나머지 항목 동적 처리
    for name, spec in SEARCH_SPACE.items():
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
                params[name] = trial.suggest_float(name, low, high, log=True)  # log scale
            else:
                params[name] = trial.suggest_float(name, low, high, step=step)

    # ── 추가 안전 제약 ──
    # min_lr < lr (혹시라도 역전되면 보정)
    if params["min_lr"] >= params["lr"]:
        params["min_lr"] = max(1e-6, params["lr"] * 0.1)

    # dropout 합 과도 방지: 세 항목 중 최댓값을 약간 줄여 완화
    if params["dropout"] + params["attn_dropout"] + params["token_dropout"] > 0.7:
        trio = ["dropout", "attn_dropout", "token_dropout"]
        biggest = max(trio, key=lambda k: params[k])
        params[biggest] = max(0.0, params[biggest] - 0.2)

    # patience <= epochs
    if params["patience"] > params["epochs"]:
        params["patience"] = max(1, params["epochs"] // 2)

    return params


def objective(trial: optuna.Trial, cli_args, csv_path):
    """Optuna 목적 함수: 학습을 완전히 조용히 실행하고 f1_macro만 반환"""
    # 전역 로깅 완전 차단
    logging.disable(logging.CRITICAL)

    try:
        params = _suggest_params(trial)

        # classifier의 기본 인자 로드
        base_args = fttransformer_classifier.parse_args()

        # CLI 인자와 trial 파라미터를 merge (우선순위: params > cli_args > base_args)
        merged = {**vars(base_args), **vars(cli_args), **params}

        # valid_size -> val_ratio 매핑(있다면)
        if "valid_size" in merged:
            merged["val_ratio"] = merged["valid_size"]

        # 최종 args
        hpo_args = argparse.Namespace(**merged)

        # stdout/stderr 완전 무음 (epoch 로그/경고 모두 억제)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            out = fttransformer_classifier.train_and_eval(
                hpo_args, device="cuda" if cli_args.use_gpu else "cpu"
            )

        if not out or "metrics" not in out:
            return 0.0

        metrics = out["metrics"]
        f1_macro = float(metrics.get("f1_macro", 0.0))
        accuracy = float(metrics.get("accuracy", 0.0))
        if not np.isfinite(f1_macro):
            f1_macro = 0.0
        if not np.isfinite(accuracy):
            accuracy = 0.0

        # CSV에 trial 결과 누적 (조용히)
        row = [trial.number] + [params[k] for k in SEARCH_SPACE.keys()] + [f1_macro, accuracy]
        with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow(row)

        return f1_macro

    except Exception:
        # 어떤 에러든 조용히 0.0 반환
        return 0.0

    finally:
        logging.disable(logging.NOTSET)  # 로깅 상태 복구


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", default="data/train.csv")
    ap.add_argument("--n_trials", type=int, default=TRIALS)
    ap.add_argument("--valid_size", type=float, default=0.2)  # classifier의 val_ratio로 매핑
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_gpu", action="store_true")
    # 필요시 test_path/target을 여기서도 덮어쓸 수 있음
    ap.add_argument("--test_path", default="data/test.csv")
    ap.add_argument("--target", default="support_needs")
    args = ap.parse_args()

    # 저장 위치 준비
    timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
    save_dir = "results/optimization_fttransformer"
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f"{timestamp}_hpo.csv")
    best_params_path = os.path.join(save_dir, f"{timestamp}_best_params.json")

    # CSV 헤더 작성
    header = ["trial"] + list(SEARCH_SPACE.keys()) + ["f1_macro", "accuracy"]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerow(header)

    # tqdm 한 줄만 사용: 진행률 + best_f1_macro 표기
    sampler = TPESampler(seed=args.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    with tqdm(total=args.n_trials, desc="Optimizing", leave=True) as pbar:
        def _cb(study_obj: optuna.Study, trial_obj: optuna.trial.FrozenTrial):
            # 진행 중엔 오직 이 한 줄만 갱신
            best = 0.0
            if study_obj.best_trial is not None and study_obj.best_trial.value is not None:
                best = study_obj.best_trial.value
            pbar.set_postfix_str(f"best_f1_macro : {best:.4f}")
            pbar.update(1)

        # 조용히 최적화 수행 (objective 내부에서 모든 출력 억제)
        study.optimize(lambda t: objective(t, args, csv_path),
                       n_trials=args.n_trials,
                       callbacks=[_cb])

    # 완료 후에만 요약 출력
    print("\n===== HPO 완료 =====")
    if not study.trials or study.best_trial is None or study.best_value is None:
        print("성공적으로 완료된 Trial이 없습니다.")
    else:
        print(f"총 Trial: {len(study.trials)}")
        print(f"최고 점수 (f1_macro): {study.best_value:.4f}")
        print("최적 파라미터:")
        for k, v in study.best_params.items():
            print(f"  - {k}: {v}")
        # 최적 파라미터 저장
        with open(best_params_path, "w", encoding="utf-8") as f:
            json.dump(study.best_params, f, indent=4, ensure_ascii=False)
        print(f"\n최적 파라미터가 '{best_params_path}'에 저장되었습니다.")
    print(f"전체 결과는 '{csv_path}'를 확인하세요.")


if __name__ == "__main__":
    main()
