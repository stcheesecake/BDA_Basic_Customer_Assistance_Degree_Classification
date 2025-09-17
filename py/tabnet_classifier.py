import argparse
import os
import random
import pandas as pd
import numpy as np
from datetime import datetime

# TabNet 및 데이터 전처리 관련 라이브러리
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.metrics import Metric
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler  # ⭐️ StandardScaler 추가
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight  # ⭐️ 클래스 가중치 계산
import torch


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class F1Macro(Metric):
    def __init__(self):
        self._name = "f1_macro"
        self._maximize = True

    def __call__(self, y_true, y_score):
        y_pred = np.argmax(y_score, axis=1)
        return f1_score(y_true, y_pred, average="macro")


def get_args():
    # ... (인자 부분은 이전과 동일) ...
    parser = argparse.ArgumentParser(description="TabNet Classifier Training Script")
    parser.add_argument("--train_path", type=str, default="data/tabnet_train.csv")
    parser.add_argument("--test_path", type=str, default="data/tabnet_test.csv")
    parser.add_argument("--target", type=str, default="support_needs")
    parser.add_argument("--submission", action='store_true')
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--n_d", type=int, default=8)
    parser.add_argument("--n_a", type=int, default=8)
    parser.add_argument("--n_steps", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=1.3)
    parser.add_argument("--cat_emb_dim", type=int, default=1)
    parser.add_argument("--n_independent", type=int, default=2)
    parser.add_argument("--n_shared", type=int, default=2)
    parser.add_argument("--mask_type", type=str, default='sparsemax')
    parser.add_argument("--lambda_sparse", type=float, default=1e-3)
    parser.add_argument("--max_epochs", type=int, default=200)  # ⭐️ 에포크 증가
    parser.add_argument("--patience", type=int, default=20)  # ⭐️ Patience 증가
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--virtual_batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=2e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation set ratio")


    args = parser.parse_args()
    return args


def preprocess_data(df_train, df_test, args):
    """데이터 전처리를 총괄하는 함수"""

    # 1. 피처 정의 및 순서 고정
    features = [col for col in df_train.columns if col not in ['ID', args.target]]
    X = df_train[features].copy()  # SettingWithCopyWarning 방지를 위해 .copy() 사용
    y = df_train[args.target]
    X = X[features]  # 컬럼 순서 고정

    # 2. 데이터 분할
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=args.val_ratio, random_state=args.seed, stratify=y
    )

    # 3. 피처 타입 정의
    ALL_CAT_FEATURES = [
        'gender', 'subscription_type', 'contract_length', 'is_older_group',
        'new_inactive', 'is_high_interaction', 'older_low_contract',
        'vip_low_interaction', 'gender_age_group', 'usage_cluster'
    ]
    categorical_cols = [col for col in ALL_CAT_FEATURES if col in X.columns]
    numerical_cols = [col for col in features if col not in categorical_cols]

    # 4. 스케일러 및 인코더 학습 (Train 데이터 기준) 및 적용
    encoders = {}
    cat_dims = []
    for col in categorical_cols:
        # 데이터 타입을 미리 변환
        X_train[col] = X_train[col].astype(str)
        X_valid[col] = X_valid[col].astype(str)

        encoder = LabelEncoder()
        X_train[col] = encoder.fit_transform(X_train[col])
        X_valid[col] = encoder.transform(X_valid[col])
        encoders[col] = encoder

        cat_dims.append(len(encoder.classes_))

    scaler = None
    if numerical_cols:
        scaler = StandardScaler()
        X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
        X_valid[numerical_cols] = scaler.transform(X_valid[numerical_cols])

    # 5. TabNet에 필요한 파라미터 계산
    cat_idxs = [X_train.columns.get_loc(col) for col in categorical_cols]

    # 6. 테스트 데이터 전처리 (존재하는 경우)
    X_test = None
    if df_test is not None:
        X_test = df_test[features].copy()
        X_test = X_test[features]  # 컬럼 순서 고정

        for col, encoder in encoders.items():
            X_test[col] = X_test[col].astype(str)
            X_test[col] = encoder.transform(X_test[col])

        if scaler and numerical_cols:
            X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    return X_train, X_valid, y_train, y_valid, X_test, cat_idxs, cat_dims

def train_and_eval(**kwargs):
    # 1. 파라미터 통합 (기존과 동일)
    args = get_args()
    args.__dict__.update(kwargs)
    seed_everything(args.seed)

    # 2. 원본 데이터 로딩
    df_train = pd.read_csv(args.train_path)
    df_test = pd.read_csv(args.test_path) if args.submission else None

    # 3. 전처리 함수 호출로 모든 데이터 준비 완료
    X_train, X_valid, y_train, y_valid, X_test, cat_idxs, cat_dims = preprocess_data(df_train, df_test, args)

    # 4. 클래스 가중치 계산
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.FloatTensor(class_weights).cuda()

    # 5. 모델 정의 (기존과 동일)
    model = TabNetClassifier(
        n_d=args.n_d, n_a=args.n_a, n_steps=args.n_steps, gamma=args.gamma,
        cat_emb_dim=args.cat_emb_dim, n_independent=args.n_independent,
        n_shared=args.n_shared, lambda_sparse=args.lambda_sparse,
        mask_type=args.mask_type, cat_idxs=cat_idxs, cat_dims=cat_dims,
        optimizer_params=dict(lr=args.learning_rate, weight_decay=args.weight_decay),
        seed=args.seed, verbose=args.verbose,
    )

    # 6. 모델 학습 (기존과 동일)
    model.fit(
        X_train=X_train.values, y_train=y_train.values,
        eval_set=[(X_valid.values, y_valid.values)],
        eval_metric=[F1Macro],
        max_epochs=args.max_epochs, patience=args.patience,
        batch_size=args.batch_size, virtual_batch_size=args.virtual_batch_size,
        weights=1, loss_fn=torch.nn.CrossEntropyLoss(weight=class_weights),
    )

    # 7. 평가 및 제출 파일 생성
    preds = model.predict(X_valid.values)
    f1 = f1_score(y_valid, preds, average="macro")
    acc = accuracy_score(y_valid, preds)

    if args.submission:
        test_preds = model.predict(X_test.values)
        submission_df = pd.DataFrame({'ID': df_test['ID'], args.target: test_preds})

        # [오류 수정] 생략되었던 파일 저장 로직 추가
        save_dir = "results/submission"
        os.makedirs(save_dir, exist_ok=True)
        now = datetime.now().strftime('%y%m%d_%H%M%S')
        filename = f"tabnet_{now}_f1_{f1:.4f}_seed_{args.seed}.csv"
        save_path = os.path.join(save_dir, filename)

        submission_df.to_csv(save_path, index=False)
        if args.verbose:
            print(f"Submission file saved to: {save_path}")

    return {'metrics': {'f1_macro': f1, 'accuracy': acc}}


if __name__ == "__main__":
    args = get_args()
    results = train_and_eval(**vars(args))

    print("\n==== Final Validation Metrics ====")
    print(f"F1 Macro : {results['metrics']['f1_macro']:.4f}")
    print(f"Accuracy : {results['metrics']['accuracy']:.4f}")