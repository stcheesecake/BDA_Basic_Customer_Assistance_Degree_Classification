#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
catboost.py  (multiclass for support_needs)
- 데이터: data/train.csv (필수), data/test.csv(선택; submission=True일 때 사용)
- 타깃: support_needs (0/1/2)
- 전처리: CatBoost의 범주형 직접 처리(스케일/원핫 불필요)
- 평가: macro-F1, accuracy, per-class 지표, 혼동행렬
- 저장: (기본) 모델/지표/로그, (옵션) submission.csv
  * 단, produce_artifacts=False로 호출되면 어떤 파일도 생성하지 않음
"""

import os
import json
import argparse
from datetime import datetime
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    f1_score, precision_recall_fscore_support, accuracy_score, confusion_matrix
)
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────────────────────────────
# 기본 하이퍼파라미터 (필요시 --params_json 또는 params_dict로 덮어쓰기)
# ─────────────────────────────────────────────────────────────────────
DEFAULT_PARAMS = dict(
    loss_function="MultiClass",
    eval_metric="TotalF1",       # macro F1 유사
    iterations=800,
    learning_rate=0.18,
    depth=7,
    l2_leaf_reg=2.0,
    random_strength=0,
    border_count=216,
    bagging_temperature=0.55,
    boosting_type="Ordered",     # GPU일 땐 자동 Plain 가드
    task_type="CPU",
    od_type="Iter",
    od_wait=100,
    random_seed=42,
    verbose=False,
    # 🔸 추가: 제출 파일 생성 여부 (True면 data/test.csv로 제출 생성)
    submission=True,
)

# ─────────────────────────────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────────────────────────────
def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def now_tag(fmt: str = "%Y%m%d_%H%M%S") -> str:
    return datetime.now().strftime(fmt)

def infer_cat_feature_indices(df: pd.DataFrame, prefer_cols: List[str] = None) -> List[int]:
    if prefer_cols:
        cols = [c for c in prefer_cols if c in df.columns]
        return [df.columns.get_loc(c) for c in cols]
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    return [df.columns.get_loc(c) for c in cat_cols]

def compute_class_weights_multiclass(y: pd.Series) -> List[float]:
    counts = y.value_counts().sort_index()
    N = len(y)
    K = counts.size
    w = (N / (K * counts)).values.astype(float)
    return w.tolist()

def to_native(o):
    import numpy as _np
    if isinstance(o, dict):
        return {k: to_native(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_native(v) for v in o]
    if isinstance(o, (_np.integer, )):
        return int(o)
    if isinstance(o, (_np.floating, )):
        return float(o)
    if isinstance(o, _np.ndarray):
        return o.tolist()
    return o

def write_report_txt(path: str, metrics: Dict, params: Dict) -> None:
    lines = []
    lines.append("==== CatBoost Multiclass Train Report ====\n")
    lines.append("[Metrics]")
    for k, v in metrics.items():
        try:
            if isinstance(v, float):
                lines.append(f"{k}: {v:.6f}")
            else:
                lines.append(f"{k}: {v}")
        except Exception:
            lines.append(f"{k}: {v}")
    lines.append("\n[Params]")
    for k in sorted(params.keys()):
        lines.append(f"{k}: {params[k]}")
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[Saved] report -> {path}")

# ─────────────────────────────────────────────────────────────────────
# 추론/제출
# ─────────────────────────────────────────────────────────────────────
def infer_and_submit(
    model: CatBoostClassifier,
    test_path: str = "data/test.csv",
    save_dir: str = "results/submission",
    target_col: str = "support_needs",
    id_candidates=("ID", "id", "Id", "index")
) -> Optional[str]:
    if not os.path.exists(test_path):
        print(f"[Skip] test csv not found: {test_path}")
        return None
    _ensure_dir(save_dir)

    test_df = pd.read_csv(test_path).copy()

    # 1) 제출용 ID 컬럼 결정
    id_col = None
    for c in id_candidates:
        if c in test_df.columns:
            id_col = c
            break
    if id_col is None:
        id_col = "ID"
        test_df[id_col] = [f"TEST_{i:05d}" for i in range(len(test_df))]

    # 2) 예측용 X_test 구성: ID/타깃 제거 (여기가 핵심 수정)
    drop_cols = [id_col, target_col]
    X_test = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns], errors="ignore")

    # 3) 범주형 지정 (필요하면 object/category 전체 자동 처리로 바꿔도 됨)
    prefer_cats = ["gender", "subscription_type"]
    cat_idx = infer_cat_feature_indices(X_test, prefer_cols=prefer_cats)

    # 4) 예측
    pred = model.predict(Pool(X_test, cat_features=cat_idx)).astype(int).ravel()

    # 5) 제출 저장
    submit = pd.DataFrame({id_col: test_df[id_col], target_col: pred})
    out_path = os.path.join(save_dir, f"catboost_{now_tag('%y%m%d_%H%M%S')}_submission.csv")
    submit.to_csv(out_path, index=False)
    print(f"[Saved] submission -> {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────
# 학습/평가
# ─────────────────────────────────────────────────────────────────────
def train_and_eval(
    train_path: str = "data/train.csv",
    target_col: str = "support_needs",
    save_dir: str = "results/catboost_optimization",
    valid_size: float = 0.2,
    seed: int = 42,
    use_gpu: bool = False,
    params_json: str = None,
    params_dict: Dict = None,
    produce_artifacts: bool = True,   # hpo.py에서 False로 넘기면 파일 저장 없음
) -> Dict:
    assert os.path.exists(train_path), f"train csv not found: {train_path}"
    if produce_artifacts:
        _ensure_dir(save_dir)

    # ── 데이터 적재 ───────────────────────────────────────────────
    df = pd.read_csv(train_path)

    # ID 제거
    drop_cols = [c for c in ["ID", "id", "Id"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # 타깃 확인
    assert target_col in df.columns, f"target '{target_col}' not in {train_path}"

    # X, y 분리
    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])

    # 범주형 강제 지정
    prefer_cats = ["gender", "subscription_type"]
    cat_idx = infer_cat_feature_indices(X, prefer_cols=prefer_cats)

    # ── 파라미터 구성 ────────────────────────────────────────────
    params = DEFAULT_PARAMS.copy()
    if use_gpu:
        params["task_type"] = "GPU"
    if params_json:
        with open(params_json, "r", encoding="utf-8") as f:
            params.update(json.load(f))
    if params_dict:
        params.update(params_dict)

    # GPU + MultiClass일 때 Ordered 금지 → Plain으로 강제 (호환 가드)
    if str(params.get("task_type", "CPU")).upper() == "GPU":
        if str(params.get("loss_function", "MultiClass")).lower().startswith("multi"):
            params["boosting_type"] = "Plain"

    # 🔴 CatBoost가 모르는 커스텀 키는 모델에 넘기면 안 됨
    #    submission 플래그만 꺼내서 내부 제출 로직에서만 사용
    submission_flag = bool(params.pop("submission", False))

    # class_weights 자동 설정(없을 때만)
    if "class_weights" not in params:
        params["class_weights"] = compute_class_weights_multiclass(y)

    # ── 데이터 분할 ───────────────────────────────────────────────
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=valid_size, stratify=y, random_state=seed
    )

    # Pool 구성
    train_pool = Pool(X_tr, y_tr, cat_features=cat_idx)
    valid_pool = Pool(X_va, y_va, cat_features=cat_idx)

    # ── 학습 ─────────────────────────────────────────────────────
    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=valid_pool, verbose=params.get("verbose", False))

    # ── 검증 지표 ────────────────────────────────────────────────
    y_pred = model.predict(valid_pool).astype(int).ravel()
    acc = accuracy_score(y_va, y_pred)
    classes_sorted = sorted(y.unique())

    prec, rec, f1, support = precision_recall_fscore_support(
        y_va, y_pred, labels=classes_sorted, average=None
    )
    f1_macro = f1_score(y_va, y_pred, average="macro")
    cm = confusion_matrix(y_va, y_pred, labels=classes_sorted)

    metrics = {
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "per_class_precision": [float(x) for x in prec],
        "per_class_recall":    [float(x) for x in rec],
        "per_class_f1":        [float(x) for x in f1],
        "per_class_support":   [int(x)   for x in support],
        "confusion_matrix":    cm.astype(int).tolist(),
        "classes":             [int(c)   for c in classes_sorted]
    }

    # ── 파일 저장 (옵션) ─────────────────────────────────────────
    if produce_artifacts:
        _ensure_dir(save_dir)
        model_path = os.path.join(save_dir, "catboost_model.cbm")
        model.save_model(model_path)
        with open(os.path.join(save_dir, "used_params.json"), "w", encoding="utf-8") as f:
            json.dump(params, f, ensure_ascii=False, indent=2)
        with open(os.path.join(save_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(to_native(metrics), f, ensure_ascii=False, indent=2)
        write_report_txt(os.path.join(save_dir, "metrics.txt"), metrics, params)

        # 제출 파일 생성 (옵션): DEFAULT_PARAMS/params_dict의 submission=True일 때만
        if submission_flag:
            infer_and_submit(model,
                             test_path="data/test.csv",
                             save_dir="results/submission",
                             target_col=target_col)

    # ── 콘솔 요약 ────────────────────────────────────────────────
    print("\n==== Summary ====")
    print(f"accuracy      : {metrics['accuracy']:.4f}")
    print(f"f1_macro      : {metrics['f1_macro']:.4f}")

    return dict(
        model=model,
        metrics=metrics,
        params=params,
        cat_idx=cat_idx
    )


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", default="data/train.csv")
    ap.add_argument("--test_path", default=None, help="(미사용) 제출은 DEFAULT_PARAMS['submission']로 제어")
    ap.add_argument("--target", default="support_needs")
    ap.add_argument("--save_dir", default="results/catboost_optimization")
    ap.add_argument("--valid_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_gpu", action="store_true")
    ap.add_argument("--params_json", default=None)
    ap.add_argument("--submission", action="store_true", help="실행 시 강제로 제출 on")
    args = ap.parse_args()

    params_dict = {}
    if args.submission:
        params_dict["submission"] = True

    train_and_eval(
        train_path=args.train_path,
        target_col=args.target,
        save_dir=args.save_dir,
        valid_size=args.valid_size,
        seed=args.seed,
        use_gpu=args.use_gpu,
        params_json=args.params_json,
        params_dict=params_dict,
        produce_artifacts=True
    )

if __name__ == "__main__":
    main()
