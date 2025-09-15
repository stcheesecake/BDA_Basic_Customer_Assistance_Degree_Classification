import argparse
import os
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils import class_weight
from datetime import datetime

import json  # 수동 가중치 파싱을 위해 추가


def get_args():
    parser = argparse.ArgumentParser()
    # --- 기존 인자 (변경 없음) ---
    parser.add_argument("--train_path", type=str, default="data/cat_train.csv")
    parser.add_argument("--test_path", type=str, default="data/cat_test.csv")
    parser.add_argument("--target", type=str, default="support_needs")
    parser.add_argument("--iterations", type=int, default=1600)
    parser.add_argument("--learning_rate", type=float, default=0.13)
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--l2_leaf_reg", type=float, default=14)
    parser.add_argument("--random_strength", type=float, default=0)
    parser.add_argument("--border_count", type=int, default=208)
    parser.add_argument("--bagging_temperature", type=float, default=0.47)
    parser.add_argument("--boosting_type", type=str, default="Ordered")
    parser.add_argument("--od_type", type=str, default="Iter")
    parser.add_argument("--od_wait", type=int, default=200)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--submission", action="store_true", default = True)
    parser.add_argument("--verbose", type=int, default=0)

    # [요청사항] class_weights를 제어할 수 있는 인자 추가
    parser.add_argument(
        "--weights",
        type=str,
        default="balanced",
        help="Class weights. 'balanced' or a JSON string like '[1.0, 2.0, 0.5]'"
    )

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

    # [요청사항] --weights 인자에 따라 가중치 계산 방식 분기
    if args.weights == "balanced":
        # 1. 'balanced' 모드 (기본값)
        class_weights = class_weight.compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        weights = dict(enumerate(class_weights))
    else:
        # 2. 수동 리스트 모드 (예: "[1, 2, 0]")
        try:
            manual_weights_list = json.loads(args.weights)
            # 클래스 레이블 0, 1, 2... 순서대로 가중치 매핑
            weights = dict(enumerate(manual_weights_list))
        except (json.JSONDecodeError, TypeError):
            print(f"경고: 잘못된 weights 형식입니다 ('{args.weights}'). 가중치를 적용하지 않습니다.")
            weights = None

    task_type = "GPU" if args.use_gpu else "CPU"
    boosting_type = "Plain" if task_type == "GPU" else args.boosting_type

    model = CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="TotalF1",
        class_weights=weights,  # 계산된 가중치 적용
        iterations=args.iterations, learning_rate=args.learning_rate, depth=args.depth,
        l2_leaf_reg=args.l2_leaf_reg, random_strength=args.random_strength, border_count=args.border_count,
        bagging_temperature=args.bagging_temperature, boosting_type=boosting_type, od_type=args.od_type,
        od_wait=args.od_wait, random_seed=args.seed, verbose=args.verbose, task_type=task_type,
    )

    model.fit(train_pool, eval_set=valid_pool)
    preds = model.predict(valid_pool)
    acc = accuracy_score(y_valid, preds)
    f1_macro = f1_score(y_valid, preds, average="macro")
    # [✅ 여기에 Submission 파일 생성 로직 추가]

    if args.submission:
        print("\nSubmission 파일을 생성합니다...")
        test_df = pd.read_csv(args.test_path)

        # Test 데이터의 피처를 Train 데이터와 동일하게 맞춰주어야 합니다.
        test_X = test_df.drop(columns=["ID"], errors="ignore")
        if 'features_to_include' in kwargs:
            test_X = test_X.reindex(columns=X.columns, fill_value=0)

        test_pool = Pool(test_X, cat_features=cat_idx)
        test_preds = model.predict(test_pool).astype(int).ravel()

        submission_df = pd.DataFrame({'ID': test_df['ID'], args.target: test_preds})

        os.makedirs("results/submission", exist_ok=True)

        # [✅ 파일 이름 동적 생성 로직으로 변경]
        now = datetime.now().strftime('%y%m%d_%H%M%S')
        submission_filename = f"catboost_{now}_submission.csv"
        submission_path = os.path.join("results/submission", submission_filename)

        submission_df.to_csv(submission_path, index=False)
        print(f"✅ Submission 파일 생성이 완료되었습니다: {submission_path}")

    return {"metrics": {"f1_macro": f1_macro, "accuracy": acc}}


if __name__ == "__main__":
    args = get_args()
    results = train_and_eval(**vars(args))
    print("\n==== Validation Metrics ====")
    print(f"Accuracy : {results['metrics']['accuracy']:.4f}")
    print(f"F1 Macro : {results['metrics']['f1_macro']:.4f}")