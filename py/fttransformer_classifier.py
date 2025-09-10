# fttransformer_classifier.py
import argparse
import os
import math
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report

# =========================================================
# 1) CLI 하이퍼파라미터 (전부 아래 로직에 매핑)
# =========================================================
def parse_args():
    p = argparse.ArgumentParser()
    # 그대로 쓰던 경로/타깃/기본 학습 설정
    p.add_argument("--train_path", type=str, default="data/train.csv")
    p.add_argument("--test_path", type=str, default="data/test.csv")
    p.add_argument("--target", type=str, default="support_needs")

    # 데이터 & 학습
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--optimizer", type=str, default="adamw",
                   choices=["adamw", "adam", "sgd"])
    p.add_argument("--sched", type=str, default="cosine",
                   choices=["cosine", "step", "none"])
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--step_size", type=int, default=10,
                   help="--sched step 일 때 사용")
    p.add_argument("--gamma", type=float, default=0.5,
                   help="--sched step 일 때 사용")

    # 손실/불균형/라벨스무딩/포컬로스
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--use_focal", action=argparse.BooleanOptionalAction, type=bool, default=True)
    p.add_argument("--focal_gamma", type=float, default=2.5)
    p.add_argument("--focal_alpha", type=float, default=0.5)

    # 모델(Transformer)
    p.add_argument("--d_model", type=int, default=512) # embed_dim
    p.add_argument("--n_heads", type=int, default=16)
    p.add_argument("--n_layers", type=int, default=6)
    p.add_argument("--ff_mult", type=int, default=8,
                   help="FFN hidden = d_model * ff_mult")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--attn_dropout", type=float, default=0.0)
    p.add_argument("--layer_norm_eps", type=float, default=1e-5)
    p.add_argument("--token_dropout", type=float, default=0.0,
                   help="열 토큰 드랍아웃 확률(0~1). 0이면 비활성")

    # 데이터 전처리/분할
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--no_stratify", action="store_true",
                   help="클래스 불균형이어도 stratify 끔")
    p.add_argument("--seed", type=int, default=42)

    # 그 외 훈련 유틸
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=10,
                   help="early stopping patience(<=0이면 비활성)")
    return p.parse_args()


# =========================================================
# 2) 유틸: 시드 고정
# =========================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# 3) 데이터 준비
# =========================================================
def prepare_tabular(df: pd.DataFrame, target_col: str):
    if target_col not in df.columns:
        raise ValueError(
            f"target_col '{target_col}'이 train에 없습니다. 현재 컬럼: {list(df.columns)}"
        )

    y_raw = df[target_col].to_numpy()
    classes, y = np.unique(y_raw, return_inverse=True)
    num_classes = int(len(classes))

    # ID/target 제거
    id_candidates = [c for c in df.columns if c.lower() in ("id", "index")]
    drop_cols = set(id_candidates + [target_col])
    feat_df = df.drop(columns=list(drop_cols), errors="ignore")

    # object → 범주형, 그 외 → 수치형
    cat_cols = [c for c in feat_df.columns if feat_df[c].dtype == "object"]
    num_cols = [c for c in feat_df.columns if c not in cat_cols]

    # 수치형 안전 변환 + 결측 0
    if len(num_cols) > 0:
        feat_df[num_cols] = feat_df[num_cols].apply(
            lambda s: pd.to_numeric(s, errors="coerce")
        ).fillna(0)

    # 범주형 코드화 (+결측 클래스)
    cat_maps, cat_codes, cat_cardinalities = {}, [], []
    for c in cat_cols:
        cat_series = feat_df[c].astype("category")
        codes = cat_series.cat.codes.to_numpy().astype(np.int64)
        codes[codes < 0] = cat_series.cat.categories.size  # 결측을 마지막 idx로
        cat_codes.append(codes)
        cat_maps[c] = list(cat_series.cat.categories) + ["<NA>"]
        cat_cardinalities.append(len(cat_maps[c]))

    # 수치형 값/표준화 통계
    num_values, num_means, num_stds = [], {}, {}
    for c in num_cols:
        v = feat_df[c].to_numpy(dtype=np.float32)
        m = float(np.nanmean(v)) if np.isnan(v).any() else float(v.mean())
        s = float(np.nanstd(v)) if np.isnan(v).any() else float(v.std() + 1e-6)
        if s == 0.0:
            s = 1.0
        v = np.nan_to_num(v, nan=m)
        num_values.append(v)
        num_means[c], num_stds[c] = m, s

    X_cat = np.stack(cat_codes, axis=1) if len(cat_codes) > 0 else np.zeros((len(df), 0), dtype=np.int64)
    X_num = np.stack(num_values, axis=1) if len(num_values) > 0 else np.zeros((len(df), 0), dtype=np.float32)

    meta = {
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "cat_cardinalities": cat_cardinalities,
        "num_means": num_means,
        "num_stds": num_stds,
        "classes": classes.tolist(),
    }
    return X_cat, X_num.astype(np.float32), y.astype(np.int64), num_classes, meta


class TabularDataset(Dataset):
    def __init__(self, X_cat, X_num, y=None):
        self.X_cat = torch.from_numpy(X_cat) if isinstance(X_cat, np.ndarray) else X_cat
        self.X_num = torch.from_numpy(X_num) if isinstance(X_num, np.ndarray) else X_num
        self.y = None if y is None else torch.from_numpy(y)

    def __len__(self):
        return self.X_cat.shape[0] if self.X_cat.numel() > 0 else self.X_num.shape[0]

    def __getitem__(self, idx):
        if self.y is None:
            return self.X_cat[idx], self.X_num[idx]
        return self.X_cat[idx], self.X_num[idx], self.y[idx]


# =========================================================
# 4) 모델: 열-토큰용 Transformer
# =========================================================
class FeatureTokenizer(nn.Module):
    def __init__(self, cat_cardinalities, num_feat_dim, d_model, token_dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.n_cat = len(cat_cardinalities)
        self.n_num = int(num_feat_dim)
        self.token_dropout = token_dropout

        self.cat_embeds = nn.ModuleList(
            [nn.Embedding(c, d_model) for c in cat_cardinalities]
        ) if self.n_cat > 0 else None

        self.num_proj = nn.ModuleList(
            [nn.Linear(1, d_model) for _ in range(self.n_num)]
        ) if self.n_num > 0 else None

        self.feature_pos = nn.Embedding(self.n_cat + self.n_num + 1, d_model)  # +1 CLS
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, x_cat: torch.Tensor, x_num: torch.Tensor):
        B = x_cat.shape[0] if x_cat.numel() > 0 else x_num.shape[0]
        tokens = []

        if self.n_cat > 0:
            for i in range(self.n_cat):
                t = self.cat_embeds[i](x_cat[:, i])  # [B, d]
                tokens.append(t.unsqueeze(1))        # [B, 1, d]

        if self.n_num > 0:
            for j in range(self.n_num):
                v = x_num[:, j:j+1]                 # [B, 1]
                t = self.num_proj[j](v)             # [B, d]
                tokens.append(t.unsqueeze(1))       # [B, 1, d]

        if len(tokens) == 0:
            raise RuntimeError("토큰이 없습니다. (범주형/수치형 열 확인)")

        x = torch.cat(tokens, dim=1)                # [B, L, d]
        L = x.shape[1]

        # (옵션) token dropout — 열 임의 드랍
        if self.training and self.token_dropout > 0.0 and L > 1:
            mask = (torch.rand(B, L, device=x.device) < self.token_dropout).unsqueeze(-1)
            x = x.masked_fill(mask, 0.0)

        # CLS + 위치 임베딩
        cls_tok = self.cls.expand(B, -1, -1)        # [B, 1, d]
        x = torch.cat([cls_tok, x], dim=1)          # [B, L+1, d]
        pos_ids = torch.arange(L + 1, device=x.device).unsqueeze(0).expand(B, -1)
        x = x + self.feature_pos(pos_ids)
        return x


class TabTransformer(nn.Module):
    def __init__(
        self,
        cat_cardinalities,
        num_feat_dim,
        d_model=128,
        n_heads=8,
        n_layers=4,
        ff_mult=4,
        dropout=0.2,
        attn_dropout=0.0,
        layer_norm_eps=1e-5,
        num_classes=2,
        token_dropout=0.0,
    ):
        super().__init__()
        self.tokenizer = FeatureTokenizer(
            cat_cardinalities, num_feat_dim, d_model, token_dropout=token_dropout
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        # PyTorch 기본 MHA dropout을 별도로 제어하고 싶을 때
        for m in enc_layer.modules():
            if isinstance(m, nn.MultiheadAttention):
                m.dropout = attn_dropout

        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.head = nn.Linear(d_model, num_classes if num_classes > 2 else 1)

    def forward(self, x_cat, x_num):
        x = self.tokenizer(x_cat, x_num)  # [B, L+1, d]
        x = self.encoder(x)               # [B, L+1, d]
        cls = self.norm(x[:, 0, :])       # [B, d]
        return self.head(cls)             # [B, C] or [B, 1]


# =========================================================
# 5) 학습 유틸
# =========================================================
def make_class_weights(y, device):
    uniq, counts = np.unique(y, return_counts=True)
    weights = counts.max() / (counts + 1e-6)
    w_tensor = torch.ones(int(uniq.max() + 1), device=device)
    for u, w in zip(uniq, weights):
        w_tensor[int(u)] = float(w)
    return w_tensor


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets, num_classes):
        if num_classes == 2:
            pt = torch.sigmoid(logits.squeeze(1))
            targets = targets.float()
            p_t = pt * targets + (1 - pt) * (1 - targets)
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = -alpha_t * (1 - p_t) ** self.gamma * torch.log(p_t + 1e-12)
        else:
            logp = F.log_softmax(logits, dim=1)
            p = torch.exp(logp)
            # one-hot
            oh = F.one_hot(targets, num_classes=num_classes).float()
            p_t = (p * oh).sum(dim=1)
            alpha_t = self.alpha * oh + (1 - self.alpha) * (1 - oh)
            alpha_t = alpha_t.sum(dim=1)
            loss = -alpha_t * (1 - p_t) ** self.gamma * logp.gather(1, targets[:, None]).squeeze(1)
        return loss.mean() if self.reduction == "mean" else loss.sum()


def build_optimizer(params, name: str, lr: float, weight_decay: float):
    name = name.lower()
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9, nesterov=True)
    raise ValueError(f"Unknown optimizer: {name}")


def cosine_with_warmup(optimizer, warmup_steps, total_steps, min_lr=1e-6):
    def fn(step):
        if step < warmup_steps:
            return max(1e-8, step / max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(min_lr, 0.5 * (1 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


def build_scheduler(optimizer, name: str, total_steps: int, warmup_ratio: float, min_lr: float,
                    step_size: int, gamma: float):
    name = name.lower()
    warmup_steps = int(warmup_ratio * total_steps)
    if name == "cosine":
        return cosine_with_warmup(optimizer, warmup_steps, total_steps, min_lr=min_lr)
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    return None


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    y_true, y_pred = [], []
    for x_cat, x_num, y in loader:
        x_cat, x_num = x_cat.to(device), x_num.to(device)
        logits = model(x_cat, x_num)
        if num_classes == 2:
            pred = (logits.squeeze(1) > 0).long().cpu().numpy()
        else:
            pred = logits.argmax(dim=1).cpu().numpy()
        y_pred.extend(pred.tolist())
        y_true.extend(y.numpy().tolist())
    if len(set(y_true)) > 1:
        f1 = f1_score(y_true, y_pred, average="macro")
    else:
        f1 = 0.0
    acc = accuracy_score(y_true, y_pred) if len(y_true) > 0 else 0.0
    return f1, acc, (y_true, y_pred)


# =========================================================
# 6) 메인 학습 루틴 (args 전부 사용)
# =========================================================
def train_and_eval(args, device):
    print("데이터를 불러오고 pytorch 기반 Tabular Transformer로 학습합니다...")

    df_train = pd.read_csv(args.train_path)
    if os.path.exists(args.test_path):
        _ = pd.read_csv(args.test_path)  # 존재 확인만

    Xc, Xn, y, num_classes, meta = prepare_tabular(df_train, args.target)

    # 수치형 표준화
    for j, c in enumerate(meta["num_cols"]):
        m, s = meta["num_means"][c], meta["num_stds"][c]
        if s == 0:
            s = 1.0
        Xn[:, j] = (Xn[:, j] - m) / s

    stratify_vec = None if args.no_stratify or len(np.unique(y)) < 2 else y
    Xc_tr, Xc_val, Xn_tr, Xn_val, y_tr, y_val = train_test_split(
        Xc, Xn, y, test_size=args.val_ratio, random_state=args.seed, stratify=stratify_vec
    )

    tr_ds = TabularDataset(Xc_tr, Xn_tr, y_tr)
    va_ds = TabularDataset(Xc_val, Xn_val, y_val)
    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    print(f"[DEBUG] num_cats={len(meta['cat_cols'])}, num_nums={len(meta['num_cols'])}, classes={num_classes}")
    print(f"[DEBUG] cat_cols={meta['cat_cols']}")
    print(f"[DEBUG] num_cols={meta['num_cols']}")

    model = TabTransformer(
        cat_cardinalities=meta["cat_cardinalities"],
        num_feat_dim=len(meta["num_cols"]),
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        ff_mult=args.ff_mult,
        dropout=args.dropout,
        attn_dropout=args.attn_dropout,
        layer_norm_eps=args.layer_norm_eps,
        num_classes=num_classes,
        token_dropout=args.token_dropout,
    ).to(device)

    optimizer = build_optimizer(model.parameters(), args.optimizer, args.lr, args.weight_decay)
    total_steps = args.epochs * max(1, len(tr_loader))
    scheduler = build_scheduler(
        optimizer, args.sched, total_steps, args.warmup_ratio, args.min_lr,
        args.step_size, args.gamma
    )

    # 손실
    if args.use_focal:
        criterion = FocalLoss(gamma=args.focal_gamma, alpha=args.focal_alpha)
    else:
        criterion = None  # 아래에서 BCE/CE로 처리

    # 클래스 가중치(멀티클래스만)
    class_weights = None
    if not args.use_focal and num_classes > 2:
        class_weights = make_class_weights(y_tr, device)

    best_f1, best_state, no_improve = -1.0, None, 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x_cat, x_num, yb in tr_loader:
            x_cat, x_num, yb = x_cat.to(device), x_num.to(device), yb.to(device)
            logits = model(x_cat, x_num)

            if criterion is not None:
                loss = criterion(logits, yb, num_classes)
            else:
                if num_classes == 2:
                    loss = F.binary_cross_entropy_with_logits(logits.squeeze(1), yb.float())
                else:
                    loss = F.cross_entropy(
                        logits, yb, weight=class_weights,
                        label_smoothing=args.label_smoothing
                    )

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            if scheduler is not None and args.sched != "step":
                scheduler.step()

            running += loss.item()

        if scheduler is not None and args.sched == "step":
            scheduler.step()

        f1, acc, _ = evaluate(model, va_loader, device, num_classes)
        print(f"Epoch {epoch}/{args.epochs} | TrainLoss {running/max(1,len(tr_loader)):.4f} | ValF1 {f1:.4f} | ValAcc {acc:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if args.patience > 0 and no_improve >= args.patience:
            print(f"Early stopping: {args.patience} epochs without improvement.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    f1, acc, (yt, yp) = evaluate(model, va_loader, device, num_classes)

    print("\nValidation Report:")
    if len(set(yt)) > 1:
        print(classification_report(yt, yp, digits=2))
        print(f"F1-macro: {f1:.4f}")
    else:
        print("단일 클래스만 감지되어 classification_report 생략")
        print(f"F1-macro: {f1:.4f}")


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    train_and_eval(args, device)


if __name__ == "__main__":
    main()
