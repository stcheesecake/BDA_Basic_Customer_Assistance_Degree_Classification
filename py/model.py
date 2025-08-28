#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
model.py
- CatBoostClassifier 학습/평가/저장/추론
- 임계값 정책:
    - THRESHOLD가 숫자면: 그 값을 고정 사용 (policy: "fixed")
    - THRESHOLD가 None이면: Balanced Accuracy(= (acc0 + acc1)/2) 최대화로 탐색 (policy: "balacc")
- 기본 하이퍼파라미터는 전역 DEFAULT_PARAMS만 사용 (best_params.json 사용 제거)
- 학습 CSV 기본: data/preprocessed_train_oof.csv
- 테스트 CSV가 주어지면 results/YYYYMMDD_submission.csv 로 저장
"""

import os
import json
import argparse
import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score, recall_score,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTENC
from joblib import dump, load
from sklearn.neighbors import NearestNeighbors

# ----------------------------- 전역 DEFAULT -----------------------------
DEFAULT_PARAMS = dict(
    iterations=2400,           # param_iterations
    learning_rate=0.1,        # param_learning_rate
    depth=9,                   # param_depth
    l2_leaf_reg=17.0,           # param_l2_leaf_reg
    border_count=80,          # param_border_count
    random_strength=1.7,       # param_random_strength
    bagging_temperature=0.25,   # param_bagging_temperature
    loss_function="Logloss",
    eval_metric="F1",
    od_type="Iter",
    od_wait=100,
    boosting_type="Ordered",
    task_type="GPU",
    random_seed=42,
    verbose=False,
    _use_smote_nc=True,  # 기본적으로 SMOTE-NC 사용
    _smote_sampling=0.85,  # 소수:다수 비율 목표 (예: 0.9 ≈ 9:10)
    _smote_k=6,  # k_neighbors
)

# ======= 임계값 제어 =======
# 숫자(예: 0.52)로 지정하면 그 임계값을 그대로 사용
# None이면 Balanced Accuracy 최대화로 임계값 탐색
THRESHOLD: Optional[float] = 0.46

# ======= EVAL POLICY (탐색 시에만 사용) =======
THRESHOLD_STRATEGY = "balacc"       # 임계값 선택 정책(탐색 시): Balanced Accuracy
THRESHOLD_GRID = np.linspace(0.05, 0.95, 181)
SCORE_KEY = "balacc"                # 로그의 대표 점수 키

# ----------------------------- argparse -----------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", default="data/preprocessed_train_oof.csv",
                    help="학습 CSV 경로(기본: data/preprocessed_train_oof.csv)")
    ap.add_argument("--test_path", default="data/preprocessed_test_oof.csv",
                    help="테스트 CSV 경로(기본: None). 지정 시 submission 생성")
    ap.add_argument("--target", default="withdrawal", help="타깃 컬럼명(기본: withdrawal)")
    ap.add_argument("--save_dir", default="results/optimization", help="모델/로그 저장 폴더")
    ap.add_argument("--valid_size", type=float, default=0.2, help="검증 비율(기본 0.2)")
    ap.add_argument("--seed", type=int, default=42, help="재현성 시드")
    ap.add_argument("--deterministic", action="store_true", help="결정론 모드(thread_count=1 권장)")
    ap.add_argument("--use_smote_nc", action="store_true", help="Train split에만 SMOTE-NC 적용")
    ap.add_argument("--smote_sampling", type=float, default=0.9, help="SMOTENC sampling_strategy (e.g., 0.8~1.0)")
    ap.add_argument("--smote_k", type=int, default=5, help="SMOTENC k_neighbors")
    return ap.parse_args()

# ----------------------------- utils -----------------------------
def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def infer_cat_feature_indices(df: pd.DataFrame) -> List[int]:
    """OOF 확률(*_oof_prob)은 제외하고, object/category만 cat_features로 지정"""
    oof_cols = [c for c in df.columns if c.endswith("_oof_prob")]
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c not in oof_cols]
    return [df.columns.get_loc(c) for c in cat_cols]

def compute_class_weights(y: pd.Series) -> List[float]:
    c0 = int((y == 0).sum())
    c1 = int((y == 1).sum())
    if c0 == 0 or c1 == 0:
        return [1.0, 1.0]
    return [1.0, c0 / c1]

def now_tag() -> str:
    return datetime.datetime.now().strftime("%Y%m%d")

def write_report_txt(path: str, params: Dict, thr: float, policy: str, metrics: Dict) -> None:
    lines = []
    lines.append("==== CatBoost Train Report ====")
    lines.append("")
    lines.append("[Hyperparameters]")
    for k in sorted(params.keys()):
        lines.append(f"{k}: {params[k]}")
    lines.append("")
    lines.append(f"[Threshold] {thr:.6f} (policy: {policy})")
    lines.append("")
    lines.append("[Metrics]")
    for k in ["f1", "auc", "precision", "recall", "acc0", "acc1", "balacc", "youden", "score"]:
        if k in metrics:
            v = metrics[k]
            try:
                lines.append(f"{k}: {v:.6f}")
            except Exception:
                lines.append(f"{k}: {v}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[Saved] report txt -> {path}")

# ----------------------- 임계값(BalAcc) 유틸 -----------------------
def _rates_at(y_true: np.ndarray, prob: np.ndarray, thr: float) -> Dict[str, float]:
    pred = (prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    acc1 = tp / (tp + fn) if (tp + fn) else 0.0  # 재현율 TPR
    acc0 = tn / (tn + fp) if (tn + fp) else 0.0  # 특이도 TNR
    balacc = 0.5 * (acc0 + acc1)
    youden = acc1 - (1.0 - acc0)
    f1 = f1_score(y_true, pred) if (tp + fp > 0 and tp + fn > 0) else 0.0
    return dict(acc0=acc0, acc1=acc1, balacc=balacc, youden=youden, f1=f1)

def find_best_threshold_balacc(y_true: np.ndarray,
                               prob: np.ndarray,
                               grid: Optional[np.ndarray] = None) -> Tuple[float, float]:
    if grid is None:
        grid = THRESHOLD_GRID
    best_thr, best_score = 0.5, -1.0
    for t in grid:
        score = _rates_at(y_true, prob, t)["balacc"]
        if score > best_score:
            best_score, best_thr = score, t
    return float(best_thr), float(best_score)

def metrics_from_cm(y_true: np.ndarray, y_pred: np.ndarray, prob: Optional[np.ndarray] = None) -> Dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    acc1 = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    balacc = 0.5 * (acc0 + acc1)
    youden = acc1 - (1.0 - acc0)
    res = dict(
        f1=f1_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred),
        recall=recall_score(y_true, y_pred),
        acc0=acc0,
        acc1=acc1,
        balacc=balacc,
        youden=youden
    )
    if prob is not None:
        try:
            res["auc"] = roc_auc_score(y_true, prob)
        except Exception:
            res["auc"] = float("nan")
    return res

def build_preprocessor(num_cols, cat_cols):
    cat_pipe = Pipeline(steps=[
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    num_pipe = Pipeline(steps=[
        ("imp", SimpleImputer(strategy="median")),
    ])
    return ColumnTransformer(
        transformers=[("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)],
        remainder="drop"
    )

# ----------------------- core training API -----------------------
def train_and_eval(
    train_path: str = "data/preprocessed_train_oof.csv",
    params: Optional[Dict] = None,  # 외부에서 dict로 덮어씌우고 싶을 때만 사용
    target_col: str = "withdrawal",
    save_dir: str = "results/optimization",
    valid_size: float = 0.2,
    seed: int = 42,
    deterministic: bool = False,
    produce_artifacts: bool = True,  # 파일 저장/생성 on/off
    quiet: bool = False,             # 콘솔 출력 on/off
) -> Dict:
    """
    단일 홀드아웃(valid_size)로 학습/평가.
    반환: {'model_path','threshold','metrics','params','cat_idx','score'}
    """
    assert os.path.exists(train_path), f"train csv not found: {train_path}"
    _ensure_dir(save_dir)

    df = pd.read_csv(train_path)
    assert target_col in df.columns, f"target '{target_col}' not in {train_path}"

    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])

    # cat_features 추정 (OOF 확률 제외 + object/category만)
    cat_idx = infer_cat_feature_indices(X)

    # 파라미터: 전역 DEFAULT_PARAMS만 사용, 필요 시 인자로 전달된 params로만 덮어쓰기
    p = DEFAULT_PARAMS.copy()
    if params is not None:
        p.update(params)

    # 결정론 옵션
    if deterministic:
        p["deterministic"] = True
        p["thread_count"] = 1

    # class_weights 자동 보정(없을 때만)
    if "class_weights" not in p and "auto_class_weights" not in p:
        p["class_weights"] = compute_class_weights(y)

    # split
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=valid_size, stratify=y, random_state=seed
    )

    # ── SMOTE-NC 분기 ─────────────────────────────────────────────
    use_smote_nc = bool(p.pop("_use_smote_nc", False))
    smote_sampling = float(p.pop("_smote_sampling", 0.9))
    smote_k = int(p.pop("_smote_k", 5))

    pre = None  # (SMOTE 경로에서만 사용)

    if use_smote_nc:
        # (a) 원본 DF 기준 열 분리
        cat_cols = X.select_dtypes(include=["object", "bool", "category"]).columns.tolist()
        num_cols = [c for c in X.columns if c not in cat_cols]

        # (b) 전처리: Train에 fit, Valid에 transform
        cat_pipe = Pipeline(steps=[
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ])
        num_pipe = Pipeline(steps=[
            ("imp", SimpleImputer(strategy="median")),
        ])
        pre = ColumnTransformer(
            transformers=[("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)],
            remainder="drop"
        )
        X_tr_t = pre.fit_transform(X_tr)
        X_va_t = pre.transform(X_va)

        # (c) 변환 결과에서 범주형 인덱스([num..., cat...])
        n_num, n_cat = len(num_cols), len(cat_cols)
        cat_idx_out = list(range(n_num, n_num + n_cat))

        # (d) Train에만 SMOTENC (누수 방지)
        smote = SMOTENC(
            categorical_features=cat_idx_out,
            sampling_strategy=smote_sampling,
            k_neighbors=smote_k,
            random_state=seed,
        )
        X_tr_bal, y_tr_bal = smote.fit_resample(X_tr_t, y_tr)

        # (e) Pool (이미 숫자화됐으므로 cat_features 불필요)
        train_pool = Pool(X_tr_bal, y_tr_bal)
        valid_pool = Pool(X_va_t, y_va)
    else:
        # 기존 경로: CatBoost가 범주형 직접 처리
        cat_idx = infer_cat_feature_indices(X)
        train_pool = Pool(X_tr, y_tr, cat_features=cat_idx)
        valid_pool = Pool(X_va, y_va, cat_features=cat_idx)

    # 학습
    model = CatBoostClassifier(**p)
    model.fit(train_pool, eval_set=valid_pool, verbose=p.get("verbose", False))

    # 임계값: 전역 THRESHOLD가 숫자면 고정 사용, 아니면 BalAcc 최대 탐색
    prob = model.predict_proba(valid_pool)[:, 1]
    if THRESHOLD is not None:
        thr = float(THRESHOLD)
        policy = "fixed"
    else:
        thr, _ = find_best_threshold_balacc(y_va.values, prob)
        policy = THRESHOLD_STRATEGY

    # 지표
    pred = (prob >= thr).astype(int)
    metrics = metrics_from_cm(y_va.values, pred, prob)
    metrics["score"] = metrics.get(SCORE_KEY, float("nan"))

    # 저장물
    model_path = os.path.join(save_dir, "catboost_model.cbm")
    if produce_artifacts:
        model.save_model(model_path)
        try:
            if pre is not None:
                dump(pre, os.path.join(save_dir, "preprocessor.joblib"))
        except Exception:
            pass
        with open(os.path.join(save_dir, "best_threshold.json"), "w", encoding="utf-8") as f:
            json.dump({"best_threshold": thr, "policy": policy}, f, ensure_ascii=False, indent=2)
        with open(os.path.join(save_dir, "used_params.json"), "w", encoding="utf-8") as f:
            json.dump(p, f, ensure_ascii=False, indent=2)
        write_report_txt(os.path.join(save_dir, "metrics.txt"), p, thr, policy, metrics)

    # 콘솔 요약
    if not quiet:
        print("\n==== Summary ====")
    if produce_artifacts:
        print(f"Model saved      : {model_path}")
    print(f"Threshold        : {thr:.4f} (policy: {policy})")
    for k in ["f1", "auc", "precision", "recall", "acc0", "acc1", "balacc", "youden", "score"]:
        if k in metrics:
            v = metrics[k]
            print(f"{k:>9}: {v:.4f}" if isinstance(v, (int, float)) else f"{k:>9}: {v}")

    return dict(
        model_path=model_path,
        threshold=thr,
        metrics=metrics,
        params=p,
        cat_idx=cat_idx,
        score=metrics["score"]
    )

# ----------------------------- inference -----------------------------
def infer_and_submit(
    model_path: str,
    threshold: float,
    test_path: str,
    target_col: str = "withdrawal",
    save_dir: str = "results",
    id_candidates: Tuple[str, ...] = ("ID", "id", "Id", "index")
) -> str:
    assert os.path.exists(model_path), f"model not found: {model_path}"
    assert os.path.exists(test_path), f"test csv not found: {test_path}"
    _ensure_dir(save_dir)

    model = CatBoostClassifier()
    model.load_model(model_path)

    test_df = pd.read_csv(test_path)
    X_test = test_df.copy()
    if target_col in X_test.columns:
        X_test = X_test.drop(columns=[target_col])

    pre_path = os.path.join(os.path.dirname(model_path), "preprocessor.joblib")
    if os.path.exists(pre_path):
        pre = load(pre_path)
        X_test_t = pre.transform(X_test)
        test_pool = Pool(X_test_t)  # numeric
    else:
        cat_idx = infer_cat_feature_indices(X_test)
        test_pool = Pool(X_test, cat_features=cat_idx)

    prob = model.predict_proba(test_pool)[:, 1]
    pred = (prob >= threshold).astype(int)

    # ID 컬럼 추정/생성
    id_col = None
    for c in id_candidates:
        if c in test_df.columns:
            id_col = c
            break
    if id_col is None:
        id_col = "ID"
        n = len(test_df)
        # 🔧 sample_submission.csv와 동일 포맷: TEST_0000 ~ TEST_0787
        test_df[id_col] = [f"TEST_{i:04d}" for i in range(n)]

    submit = pd.DataFrame({id_col: test_df[id_col], target_col: pred})
    out_path = os.path.join(save_dir, f"{now_tag()}_submission.csv")
    submit.to_csv(out_path, index=False)
    print(f"[Saved] submission -> {out_path}")
    return out_path

# ------------------------------- CLI --------------------------------
def main_cli():
    args = parse_args()
    out = train_and_eval(
        train_path=args.train_path,
        params=None,  # 외부에서 덮어쓸 필요가 있으면 dict로 전달
        target_col=args.target,
        save_dir=args.save_dir,
        valid_size=args.valid_size,
        seed=args.seed,
        deterministic=args.deterministic,
    )
    if args.test_path is not None and str(args.test_path).lower() != "none":
        infer_and_submit(
            model_path=out["model_path"],
            threshold=out["threshold"],
            test_path=args.test_path,
            target_col=args.target,
            save_dir="results"
        )

if __name__ == "__main__":
    main_cli()
