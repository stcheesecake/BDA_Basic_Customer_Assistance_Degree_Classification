import argparse
import os
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils import class_weight
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score, average_precision_score
)
from sklearn.preprocessing import label_binarize
from datetime import datetime

import json  # 수동 가중치 파싱을 위해 추가


def get_args():
    parser = argparse.ArgumentParser()
    # --- 기존 인자 (변경 없음) ---
    parser.add_argument("--train_path", type=str, default="data/cattest_train.csv")
    parser.add_argument("--test_path", type=str, default="data/cattest_test.csv")
    parser.add_argument("--target", type=str, default="support_needs")
    parser.add_argument("--iterations", type=int, default=1400)
    parser.add_argument("--learning_rate", type=float, default=0.015)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--l2_leaf_reg", type=float, default=18)
    parser.add_argument("--border_count", type=int, default=240)
    parser.add_argument("--random_strength", type=float, default=0.23)
    parser.add_argument("--bagging_temperature", type=float, default=0.37)
    parser.add_argument("--boosting_type", type=str, default="Ordered")
    parser.add_argument("--od_type", type=str, default="Iter")
    parser.add_argument("--od_wait", type=int, default=200)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--submission", action="store_true", default = False)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--metrics", type=str, default="f1_macro",
        help="평가 지표 선택: f1_macro, class0, class1, class2, bal_acc, auprc_macro, acc"
    )

    # [요청사항] class_weights를 제어할 수 있는 인자 추가
    parser.add_argument(
        "--weights",
        type=str,
        default="[2.85, 5.06, 4.4]",
        help="Class weights. 'balanced' or a JSON string like '[1.0, 2.0, 0.5]'"
    )

    args = parser.parse_args(args=None if __name__ == "__main__" else [])
    return args


def train_and_eval(**kwargs):
    args = get_args()
    args.__dict__.update(kwargs)

    df = pd.read_csv(args.train_path)

    X = df.drop(columns=[args.target, "ID"], errors="ignore")
    y = df[args.target]

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
    preds = model.predict(valid_pool).astype(int).ravel()
    probas = model.predict_proba(valid_pool)

    # ---- 기본 메트릭 계산 ----
    acc = accuracy_score(y_valid, preds)
    f1_macro = f1_score(y_valid, preds, average="macro")
    f1_classes = f1_score(y_valid, preds, average=None, labels=[0, 1, 2])
    f1_class0, f1_class1, f1_class2 = f1_classes[0], f1_classes[1], f1_classes[2]
    bal_acc = balanced_accuracy_score(y_valid, preds)

    try:
        y_bin = label_binarize(y_valid, classes=[0, 1, 2])
        auprcs = [average_precision_score(y_bin[:, i], probas[:, i]) for i in range(y_bin.shape[1])]
        auprc_macro = float(np.mean(auprcs))
    except Exception:
        auprc_macro = None

    # ---- metric 선택 ----
    if args.metrics == "f1_macro":
        score = f1_macro
    elif args.metrics == "class0":
        score = f1_class0
    elif args.metrics == "class1":
        score = f1_class1
    elif args.metrics == "class2":
        score = f1_class2
    elif args.metrics == "bal_acc":
        score = bal_acc
    elif args.metrics == "auprc_macro":
        score = auprc_macro if auprc_macro is not None else f1_macro
    elif args.metrics == "acc":
        score = acc
    else:
        score = f1_macro  # fallback

    metrics = {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_class0": f1_class0,
        "f1_class1": f1_class1,
        "f1_class2": f1_class2,
        "balanced_accuracy": bal_acc,
        "auprc_macro": auprc_macro,
        "chosen_metric": args.metrics,
        "score": score,
    }

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

    return {"metrics": metrics}


if __name__ == "__main__":
    args = get_args()
    results = train_and_eval(**vars(args))
    print("\n==== Validation Metrics ====")
    print(f"Accuracy : {results['metrics']['accuracy']:.4f}")
    print(f"F1 Macro : {results['metrics']['f1_macro']:.4f}")