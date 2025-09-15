import argparse
import os
import json
import logging
from datetime import datetime
from typing import Dict

import joblib
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    f1_score, precision_recall_fscore_support, accuracy_score, confusion_matrix
)
from sklearn.model_selection import train_test_split


# ======================================================
# Argument Parser (DEFAULT_PARAMS 대체)
# ======================================================
def get_args():
    parser = argparse.ArgumentParser()

    # 데이터 / 경로
    parser.add_argument("--train_path", type=str, default="data/train.csv")
    parser.add_argument("--test_path", type=str, default="data/test.csv")
    parser.add_argument("--target", type=str, default="support_needs")
    parser.add_argument("--save_dir", type=str, default="results/lightgbm_optimization")

    # 학습 관련
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid_size", type=float, default=0.2)
    parser.add_argument("--n_estimators", type=int, default=1600)
    parser.add_argument("--learning_rate", type=float, default=0.0218)
    parser.add_argument("--num_leaves", type=int, default=159)
    parser.add_argument("--max_depth", type=int, default=-23)
    parser.add_argument("--min_child_samples", type=int, default=100)
    parser.add_argument("--subsample", type=float, default=1.0)
    parser.add_argument("--colsample_bytree", type=float, default=1.0)
    parser.add_argument("--reg_alpha", type=float, default=1.1097)
    parser.add_argument("--reg_lambda", type=float, default=0.0241)

    # 실행 옵션
    parser.add_argument("--use_gpu", action="store_true", default=True)
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--verbosity", type=int, default=-1)
    parser.add_argument("--produce_artifacts", action="store_true")
    parser.add_argument("--submission", action="store_true")

    return parser.parse_args()


# ======================================================
# 학습 및 평가 함수
# ======================================================
def train_and_eval(**kwargs):
    args = get_args()
    for k, v in kwargs.items():
        if hasattr(args, k):
            setattr(args, k, v)
    os.makedirs(args.save_dir, exist_ok=True)

    # ------------------ 1. 파라미터 ------------------
    params = {
        "objective": "multiclass",
        "metric": "multi_logloss",
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "max_depth": args.max_depth,
        "min_child_samples": args.min_child_samples,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "reg_alpha": args.reg_alpha,
        "reg_lambda": args.reg_lambda,
        "random_state": args.seed,
        "n_jobs": args.n_jobs,
        "verbose": args.verbosity,
    }

    if args.use_gpu:
        params["device"] = "gpu"

    # ------------------ 2. 로거 ------------------
    logger = logging.getLogger(__name__)
    if args.produce_artifacts:
        log_path = os.path.join(args.save_dir, "run.log")
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s",
                            filename=log_path, filemode="w")
        logger.addHandler(logging.StreamHandler())
    else:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

    # ------------------ 3. 데이터 ------------------
    logger.info("데이터를 불러옵니다...")
    df = pd.read_csv(args.train_path)

    drop_cols = [c for c in ["ID", "id", "Id"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    X = df.drop(columns=[args.target])
    y = df[args.target]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=args.valid_size, random_state=args.seed, stratify=y
    )
    logger.info(f"훈련 데이터 형태: {X_train.shape}, 검증 데이터 형태: {X_valid.shape}")

    # ------------------ 4. 학습 ------------------
    logger.info("모델 학습을 시작합니다...")
    model = lgb.LGBMClassifier(**params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=[lgb.early_stopping(100, verbose=False)]
    )
    logger.info("모델 학습 완료.")

    # ------------------ 5. 평가 ------------------
    logger.info("모델 평가를 시작합니다...")
    preds = model.predict(X_valid)
    metrics = _calculate_metrics(y_valid, preds)

    # ------------------ 6. 결과 저장 ------------------
    if args.produce_artifacts:
        logger.info(f"결과를 '{args.save_dir}'에 저장합니다.")
        joblib.dump(model, os.path.join(args.save_dir, "model.joblib"))
        with open(os.path.join(args.save_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)
        with open(os.path.join(args.save_dir, "params.json"), "w") as f:
            json.dump(params, f, indent=4)
        _save_confusion_matrix(y_valid, preds, args.save_dir)
        logger.info("결과 저장 완료.")

    # ------------------ 7. 제출 파일 ------------------
    if args.submission:
        logger.info("제출 파일 생성을 시작합니다...")
        if args.test_path is None or not os.path.exists(args.test_path):
            logger.error(f"제출 모드이지만 test_path가 제공되지 않았거나 파일이 없습니다: {args.test_path}")
        else:
            test_df = pd.read_csv(args.test_path)
            test_ids = test_df["ID"]

            test_drop_cols = [c for c in ["ID", "id", "Id"] if c in test_df.columns]
            if test_drop_cols:
                test_df = test_df.drop(columns=test_drop_cols)

            test_df = pd.get_dummies(test_df, columns=categorical_cols, drop_first=True)

            # train/test 칼럼 맞추기
            missing_in_test = set(X.columns) - set(test_df.columns)
            for c in missing_in_test:
                test_df[c] = 0
            extra_in_test = set(test_df.columns) - set(X.columns)
            test_df = test_df.drop(columns=list(extra_in_test))
            test_df = test_df[X.columns]

            test_preds = model.predict(test_df)
            submission_df = pd.DataFrame({"ID": test_ids, args.target: test_preds})

            timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
            submission_filename = f"lightgbm_{timestamp}_submission.csv"
            submission_path = os.path.join(args.save_dir, submission_filename)

            submission_df.to_csv(submission_path, index=False)
            logger.info(f"제출 파일이 '{submission_path}'에 저장되었습니다.")

    logger.info("\n===== 평가 결과 =====")
    logger.info(f"accuracy      : {metrics['accuracy']:.4f}")
    logger.info(f"f1_macro      : {metrics['f1_macro']:.4f}")

    return {"model": model, "metrics": metrics, "params": params}


def _calculate_metrics(y_true, y_pred) -> Dict:
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")

    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average=None, labels=[0, 1, 2])
    per_class_metrics = {
        f"class_{i}": {"precision": p[i], "recall": r[i], "f1_score": f[i]}
        for i in range(len(p))
    }

    return {"accuracy": accuracy, "f1_macro": f1, "per_class": per_class_metrics}


def _save_confusion_matrix(y_true, y_pred, save_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    plt.close()


# ======================================================
# 실행부
# ======================================================
if __name__ == "__main__":
    result = train_and_eval()
    print(result["metrics"])


def DEFAULT_PARAMS():
    return None