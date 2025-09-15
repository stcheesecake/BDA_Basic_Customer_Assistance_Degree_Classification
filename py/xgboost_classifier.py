import argparse
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report


# ======================================================
# Argument Parser (DEFAULT_PARAMS 대체)
# ======================================================
def get_args():
    parser = argparse.ArgumentParser()

    # 데이터 / 경로
    parser.add_argument("--train_path", type=str, default="data/train.csv")
    parser.add_argument("--test_path", type=str, default=None)
    parser.add_argument("--target", type=str, default="support_needs")
    parser.add_argument("--save_dir", type=str, default="results/xgboost_optimization")

    # 학습 관련
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid_size", type=float, default=0.2)
    parser.add_argument("--n_estimators", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=0.0369)
    parser.add_argument("--max_depth", type=int, default=18)
    parser.add_argument("--subsample", type=float, default=1.0)
    parser.add_argument("--colsample_bytree", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.0520)
    parser.add_argument("--reg_alpha", type=float, default=2.22)
    parser.add_argument("--reg_lambda", type=float, default=0.0033)

    # 실행 옵션
    parser.add_argument("--use_gpu", action="store_true")   # GPU 사용 여부
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--early_stopping_rounds", type=int, default=100)
    parser.add_argument("--verbosity", type=int, default=0)
    parser.add_argument("--produce_artifacts", action="store_true")  # 로그/결과물 저장 여부

    return parser.parse_args()


# ======================================================
# 학습 및 평가 함수
# ======================================================
def train_and_eval(**kwargs):
    args = get_args()

    # feature_experiments.py에서 넘어온 인자들을 덮어쓰기
    for k, v in kwargs.items():
        if hasattr(args, k):
            setattr(args, k, v)

    os.makedirs(args.save_dir, exist_ok=True)
    seed = args.seed

    # 데이터 로드
    df = pd.read_csv(args.train_path)
    y = df[args.target]

    # ID 컬럼 제거
    drop_cols = [c for c in ["ID", "id", "Id"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    X = df.drop(columns=[args.target])

    # 문자열 컬럼은 원-핫 인코딩
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    if categorical_cols:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # train/valid split
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=args.valid_size, random_state=seed, stratify=y
    )

    # XGBoost 파라미터
    params = {
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "max_depth": args.max_depth,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "gamma": args.gamma,
        "reg_alpha": args.reg_alpha,
        "reg_lambda": args.reg_lambda,
        "random_state": seed,
        "seed": seed,
        "n_jobs": args.n_jobs,
        "verbosity": args.verbosity,
    }

    if args.use_gpu:
        params["tree_method"] = "gpu_hist"
        params["predictor"] = "gpu_predictor"
    else:
        # CPU 병렬 학습 강제
        params["tree_method"] = "hist"  # 병렬화된 히스토그램 알고리즘
        params["predictor"] = "cpu_predictor"

    # 모델 정의
    model = xgb.XGBClassifier(**params)

    # 학습
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=args.verbosity,
        early_stopping_rounds=args.early_stopping_rounds
    )

    # 평가
    y_pred = model.predict(X_valid)
    acc = accuracy_score(y_valid, y_pred)
    f1 = f1_score(y_valid, y_pred, average="macro")
    metrics = {
        "accuracy": acc,
        "f1_macro": f1,
        "classification_report": classification_report(y_valid, y_pred, output_dict=True)
    }

    # ======================================================
    # 산출물 저장 (옵션 produce_artifacts)
    # ======================================================
    if args.produce_artifacts:
        # 로그 파일
        with open(os.path.join(args.save_dir, "run.log"), "w", encoding="utf-8") as f:
            f.write(f"Seed: {seed}\n")
            f.write(f"Params: {json.dumps(params, indent=2)}\n")
            f.write(f"Accuracy: {acc:.4f}, F1_macro: {f1:.4f}\n")

        # metrics 저장
        with open(os.path.join(args.save_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        # params 저장
        with open(os.path.join(args.save_dir, "params.json"), "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2)

        # confusion matrix 저장
        cm = confusion_matrix(y_valid, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.savefig(os.path.join(args.save_dir, "confusion_matrix.png"))
        plt.close()

        # submission 파일 저장 (test_path 있으면)
        if args.test_path:
            test_df = pd.read_csv(args.test_path)
            # 동일하게 전처리 적용
            if drop_cols:
                test_df = test_df.drop(columns=drop_cols)
            if categorical_cols:
                test_df = pd.get_dummies(test_df, columns=categorical_cols, drop_first=True)
                # train/test 칼럼 불일치 맞추기
                test_df = test_df.reindex(columns=X.columns, fill_value=0)
            preds = model.predict(test_df)
            sub = pd.DataFrame({args.target: preds})
            sub.to_csv(os.path.join(args.save_dir, "submission.csv"), index=False)

    return {"model": model, "metrics": metrics}


# ======================================================
# 실행부
# ======================================================
if __name__ == "__main__":
    result = train_and_eval()
    print(result["metrics"])
