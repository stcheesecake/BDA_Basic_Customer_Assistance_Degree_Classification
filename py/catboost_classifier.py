import argparse
import os
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils import class_weight


def get_args():
    parser = argparse.ArgumentParser()
    # --- 원본.py의 모든 기본값과 완벽히 일치 ---
    parser.add_argument("--train_path", type=str, default="data/2_train.csv")
    parser.add_argument("--test_path", type=str, default="data/2_test.csv")
    parser.add_argument("--target", type=str, default="support_needs")
    parser.add_argument("--iterations", type=int, default=1600)
    parser.add_argument("--learning_rate", type=float, default=0.13)
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--l2_leaf_reg", type=float, default=14.0)
    parser.add_argument("--random_strength", type=float, default=0)
    parser.add_argument("--border_count", type=int, default=208)
    parser.add_argument("--bagging_temperature", type=float, default=0.7)
    parser.add_argument("--boosting_type", type=str, default="Ordered")
    parser.add_argument("--od_type", type=str, default="Iter")
    parser.add_argument("--od_wait", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--submission", action="store_true")
    parser.add_argument("--verbose", type=int, default=0)

    args = parser.parse_args(args=None if __name__ == "__main__" else [])
    return args


def train_and_eval(**kwargs):
    args = get_args()
    args.__dict__.update(kwargs)

    df = pd.read_csv(args.train_path)

    if 'features_to_include' in kwargs:
        base_features = ['age', 'gender', 'tenure', 'frequent', 'payment_interval', 'subscription_type',
                         'contract_length', 'after_interaction']
        features = base_features + args.features_to_include
        X = df[features]
    else:
        X = df.drop(columns=[args.target, "ID"], errors="ignore")

    y = df[args.target]
    cat_features = X.select_dtypes(include=["object", "category"]).columns
    cat_idx = [X.columns.get_loc(col) for col in cat_features]
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )
    train_pool = Pool(X_train, y_train, cat_features=cat_idx)
    valid_pool = Pool(X_valid, y_valid, cat_features=cat_idx)

    weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    weights = dict(enumerate(weights))

    task_type = "GPU" if args.use_gpu else "CPU"
    boosting_type = "Plain" if task_type == "GPU" else args.boosting_type

    model = CatBoostClassifier(
        loss_function="MultiClass", eval_metric="TotalF1", class_weights=weights,
        iterations=args.iterations, learning_rate=args.learning_rate, depth=args.depth,
        l2_leaf_reg=args.l2_leaf_reg, random_strength=args.random_strength, border_count=args.border_count,
        bagging_temperature=args.bagging_temperature, boosting_type=boosting_type, od_type=args.od_type,
        od_wait=args.od_wait, random_seed=args.seed, verbose=args.verbose, task_type=task_type,
    )

    model.fit(train_pool, eval_set=valid_pool)
    preds = model.predict(valid_pool)
    acc = accuracy_score(y_valid, preds)
    f1_macro = f1_score(y_valid, preds, average="macro")

    # [핵심 수정] 반환 형식을 feature_experiments.py에 맞게 이중 구조로 변경
    return {"metrics": {"f1_macro": f1_macro, "accuracy": acc}}


if __name__ == "__main__":
    args = get_args()
    results = train_and_eval(**vars(args))
    # 단독 실행 시에는 내부 값에 접근하여 출력
    print("\n==== Validation Metrics ====")
    print(f"Accuracy : {results['metrics']['accuracy']:.4f}")
    print(f"F1 Macro : {results['metrics']['f1_macro']:.4f}")