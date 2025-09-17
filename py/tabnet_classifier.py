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
    parser.add_argument("--train_path", type=str, default="data/cat_train.csv")
    parser.add_argument("--test_path", type=str, default="data/cat_test.csv")
    parser.add_argument("--target", type=str, default="support_needs")
    parser.add_argument("--submission", action='store_true')
    parser.add_argument("--seed", type=int, default=42)
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

    args = parser.parse_args()
    return args


def train_and_eval(**kwargs):
    # ⭐️ [수정] kwargs에서 'train_path'를 받아 데이터프레임을 직접 읽어옵니다.
    df_train = pd.read_csv(kwargs['train_path'])

    seed = kwargs.get("seed", 42)
    verbose = kwargs.get("verbose", 1)
    seed_everything(seed)

    target_col = kwargs.get("target", "support_needs")

    # --- 전처리 ---
    features = [col for col in df_train.columns if col not in ['ID', target_col]]

    ALL_CAT_FEATURES = [
        'gender', 'subscription_type', 'contract_length', 'is_older_group',
        'new_inactive', 'is_high_interaction', 'older_low_contract',
        'vip_low_interaction', 'gender_age_group', 'usage_cluster'
    ]

    categorical_cols = [col for col in ALL_CAT_FEATURES if col in df_train.columns]
    numerical_cols = [col for col in features if col not in categorical_cols]

    encoders = {}
    for col in categorical_cols:
        df_train[col] = df_train[col].astype(str)
        encoder = LabelEncoder()
        df_train[col] = encoder.fit_transform(df_train[col])
        encoders[col] = encoder

    scaler = None
    if numerical_cols:
        scaler = StandardScaler()
        df_train[numerical_cols] = scaler.fit_transform(df_train[numerical_cols])

    X = df_train[features]
    y = df_train[target_col]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=kwargs.get("val_ratio", 0.2), random_state=seed, stratify=y
    )

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.FloatTensor(class_weights).cuda()

    cat_idxs = [X.columns.get_loc(col) for col in categorical_cols]
    cat_dims = [df_train[col].nunique() for col in categorical_cols]

    # --- 모델 학습 ---
    model = TabNetClassifier(
        n_d=kwargs.get("n_d", 8),
        n_a=kwargs.get("n_a", 8),
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
        optimizer_params=dict(lr=kwargs.get("lr", 2e-2)),
        seed=seed,
        verbose=verbose,
    )

    model.fit(
        X_train=X_train.values, y_train=y_train.values,
        eval_set=[(X_valid.values, y_valid.values)],
        eval_metric=[F1Macro],
        max_epochs=kwargs.get("epochs", 100),
        patience=kwargs.get("patience", 15),
        batch_size=kwargs.get("batch_size", 1024),
        weights=1,
        loss_fn=torch.nn.CrossEntropyLoss(weight=class_weights),
    )

    preds = model.predict(X_valid.values)
    f1 = f1_score(y_valid, preds, average="macro")
    acc = accuracy_score(y_valid, preds)

    if verbose:
        print(f"TabNet Validation F1 Macro: {f1:.4f}, Accuracy: {acc:.4f}")

    # --- Submission 파일 생성 로직 ---
    if kwargs.get("submission"):
        if verbose:
            print("Generating submission file...")

        test_path = kwargs.get("test_path")
        if not test_path:
            raise ValueError("Submission requires a 'test_path' argument.")

        df_test = pd.read_csv(test_path)

        # 테스트 데이터 전처리 (Train에서 학습한 Encoder/Scaler 사용)
        for col, encoder in encoders.items():
            df_test[col] = df_test[col].astype(str)
            df_test[col] = encoder.transform(df_test[col])

        if scaler and numerical_cols:
            df_test[numerical_cols] = scaler.transform(df_test[numerical_cols])

        test_preds = model.predict(df_test[features].values)

        # CSV 파일 생성
        submission_df = pd.DataFrame({'ID': df_test['ID'], target_col: test_preds})

        save_dir = "results/submission"
        os.makedirs(save_dir, exist_ok=True)
        now = datetime.now().strftime('%y%m%d_%H%M%S')

        # 파일명 형식 (catboost_classifier.py와 유사하게)
        filename = f"tabnet_{now}_f1_{f1:.4f}_seed_{seed}.csv"
        save_path = os.path.join(save_dir, filename)

        submission_df.to_csv(save_path, index=False)
        if verbose:
            print(f"Submission file saved to: {save_path}")

    return {'metrics': {'f1_macro': f1, 'accuracy': acc}}


if __name__ == "__main__":
    args = get_args()
    kwargs = vars(args)

    print(f"Loading data from: {kwargs['train_path']}")
    df = pd.read_csv(kwargs['train_path'])
    kwargs['df_train'] = df

    train_and_eval(**kwargs)