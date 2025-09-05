#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
lightgbm_classifier.py  (multiclass for support_needs)
- 데이터: data/train.csv (필수), data/test.csv(선택; submission=True일 때 사용)
- 타깃: support_needs (0/1/2)
- 전처리: 범주형 피처에 대한 원-핫 인코딩 적용
- 평가: macro-F1, accuracy, per-class 지표, 혼동행렬
- 저장: (기본) 모델/지표/로그, (옵션) lightgbm_YYMMDD_HHMMSS_submission.csv
  * 단, produce_artifacts=False로 호출되면 어떤 파일도 생성하지 않음
"""

import os
import json
import argparse
import logging
from datetime import datetime
from typing import List, Dict, Optional

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    f1_score, precision_recall_fscore_support, accuracy_score, confusion_matrix
)
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────────────────────────────
# 기본 하이퍼파라미터
# ─────────────────────────────────────────────────────────────────────
DEFAULT_PARAMS = dict(
    objective="multiclass",
    metric="multi_logloss",
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    submission=False,
)


# ─────────────────────────────────────────────────────────────────────
# 모델 학습 및 평가
# ─────────────────────────────────────────────────────────────────────
def train_and_eval(
        train_path: str,
        test_path: Optional[str] = None,
        target_col: str = "support_needs",
        save_dir: Optional[str] = "results/default",
        valid_size: float = 0.2,
        seed: int = 42,
        use_gpu: bool = False,
        params_dict: Optional[Dict] = None,
        produce_artifacts: bool = True
):
    """
    LightGBM 모델 학습, 평가 및 결과 저장을 수행합니다.
    """
    # ------------------ 1. 파라미터 및 로거 준비 ------------------
    params = DEFAULT_PARAMS.copy()
    if params_dict:
        params.update(params_dict)

    submission_mode = params.pop("submission", False)

    if use_gpu:
        params['device'] = 'gpu'

    logger = logging.getLogger(__name__)
    if produce_artifacts:
        os.makedirs(save_dir, exist_ok=True)
        log_path = os.path.join(save_dir, "run.log")
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', filename=log_path, filemode='w')
        logger.addHandler(logging.StreamHandler())
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    # ------------------ 2. 데이터 준비 ------------------
    logger.info("데이터를 불러옵니다...")
    df = pd.read_csv(train_path)

    drop_cols = [c for c in ["ID", "id", "Id"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=valid_size, random_state=seed, stratify=y
    )
    logger.info(f"훈련 데이터 형태: {X_train.shape}, 검증 데이터 형태: {X_valid.shape}")

    # ------------------ 3. 모델 학습 ------------------
    logger.info("모델 학습을 시작합니다...")
    model = lgb.LGBMClassifier(**params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=[lgb.early_stopping(100, verbose=False)]
    )
    logger.info("모델 학습 완료.")

    # ------------------ 4. 평가 ------------------
    logger.info("모델 평가를 시작합니다...")
    preds = model.predict(X_valid)
    metrics = _calculate_metrics(y_valid, preds)

    # ------------------ 5. 결과 저장 (옵션) ------------------
    if produce_artifacts:
        logger.info(f"결과를 '{save_dir}'에 저장합니다.")
        joblib.dump(model, os.path.join(save_dir, "model.joblib"))
        with open(os.path.join(save_dir, "metrics.json"), 'w') as f:
            json.dump(metrics, f, indent=4)
        params_to_save = {k: str(v) for k, v in params.items()}
        with open(os.path.join(save_dir, "params.json"), 'w') as f:
            json.dump(params_to_save, f, indent=4)
        _save_confusion_matrix(y_valid, preds, save_dir)
        logger.info("결과 저장 완료.")

    # ------------------ 6. 제출 파일 생성 (옵션) ------------------
    if submission_mode:
        logger.info("제출 파일 생성을 시작합니다...")
        if test_path is None or not os.path.exists(test_path):
            logger.error(f"제출 모드이지만 test_path가 제공되지 않았거나 파일이 없습니다: {test_path}")
        else:
            test_df = pd.read_csv(test_path)
            test_ids = test_df["ID"]

            test_drop_cols = [c for c in ["ID", "id", "Id"] if c in test_df.columns]
            if test_drop_cols:
                test_df = test_df.drop(columns=test_drop_cols)

            test_df = pd.get_dummies(test_df, columns=categorical_cols, drop_first=True)

            train_cols = X.columns
            test_cols = test_df.columns
            missing_in_test = set(train_cols) - set(test_cols)
            for c in missing_in_test:
                test_df[c] = 0
            extra_in_test = set(test_cols) - set(train_cols)
            test_df = test_df.drop(columns=list(extra_in_test))
            test_df = test_df[train_cols]

            test_preds = model.predict(test_df)
            submission_df = pd.DataFrame({"ID": test_ids, target_col: test_preds})

            # [수정] 파일 이름 형식 변경
            timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
            submission_filename = f"lightgbm_{timestamp}_submission.csv"
            submission_path = os.path.join(save_dir, submission_filename)

            submission_df.to_csv(submission_path, index=False)
            logger.info(f"제출 파일이 '{submission_path}'에 저장되었습니다.")

    logger.info("\n===== 평가 결과 =====")
    logger.info(f"accuracy      : {metrics['accuracy']:.4f}")
    logger.info(f"f1_macro      : {metrics['f1_macro']:.4f}")

    return dict(model=model, metrics=metrics, params=params)


def _calculate_metrics(y_true, y_pred) -> Dict:
    """평가 지표를 계산합니다."""
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")

    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average=None, labels=[0, 1, 2])
    per_class_metrics = {
        f"class_{i}": {"precision": p[i], "recall": r[i], "f1_score": f[i]}
        for i in range(len(p))
    }

    return {
        "accuracy": accuracy,
        "f1_macro": f1,
        "per_class": per_class_metrics
    }


def _save_confusion_matrix(y_true, y_pred, save_dir):
    """혼동 행렬을 계산하고 이미지 파일로 저장합니다."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    plt.close()


# ─────────────────────────────────────────────────────────────────────
# CLI (Command Line Interface)
# ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", default="data/train.csv")
    ap.add_argument("--test_path", default="data/test.csv")
    ap.add_argument("--target", default="support_needs")
    ap.add_argument("--save_dir", default="results/lightgbm_optimization")
    ap.add_argument("--valid_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_gpu", action="store_true")
    ap.add_argument("--params_json", default=None)
    ap.add_argument("--submission", action="store_true")
    args = ap.parse_args()

    params_dict = {}
    if args.params_json:
        with open(args.params_json, 'r') as f:
            params_dict = json.load(f)
    if args.submission:
        params_dict['submission'] = True

    train_and_eval(
        train_path=args.train_path,
        test_path=args.test_path,
        target_col=args.target,
        save_dir=args.save_dir,
        valid_size=args.valid_size,
        seed=args.seed,
        use_gpu=args.use_gpu,
        params_dict=params_dict,
        produce_artifacts=True
    )


if __name__ == "__main__":
    main()