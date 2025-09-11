import os
import json
import argparse
import logging
from datetime import datetime

import math
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.transformer")

# csv 기록 스위치
LOG_TO_CSV: bool = True           # 기록 끄려면 False
CSV_LOG_PATH: str = "ft_transformer_temp.csv"

# =========================================================
# 1) CLI 하이퍼파라미터
# =========================================================
def parse_args():
    p = argparse.ArgumentParser()
    # 데이터
    p.add_argument("--train_path", type=str, default="data/train.csv")
    p.add_argument("--test_path", type=str, default="data/test.csv")
    p.add_argument("--target", type=str, default="support_needs")

    # 학습 설정
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=3e-4)
    p.add_argument("--optimizer", type=str, default="adamw",
                   choices=["adamw", "adam", "sgd"])
    p.add_argument("--sched", type=str, default="cosine",
                   choices=["cosine", "step", "none"])
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--step_size", type=int, default=10)
    p.add_argument("--gamma", type=float, default=0.1)

    # 손실
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--use_focal", action=argparse.BooleanOptionalAction, type=bool, default=True)
    p.add_argument("--focal_gamma", type=float, default=2.1)
    p.add_argument("--focal_alpha", type=float, default=0.98)
    p.add_argument("--alpha_vec", type=str, default="1.0, 1.0, 1.00", help="예: '0.65,1.20,0.85' (클래스 수와 길이가 같아야 함)")

    # 모델 구조
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--n_layers", type=int, default=12)
    p.add_argument("--ff_mult", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--attn_dropout", type=float, default=0.1)
    p.add_argument("--layer_norm_eps", type=float, default=1e-5)
    p.add_argument("--token_dropout", type=float, default=0.1)

    # 기타
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--no_stratify", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=15)

    # 저장 관련
    p.add_argument("--save_dir", type=str, default="results/fttransformer_default")
    p.add_argument("--produce_artifacts", action="store_true")

    return p.parse_args()


# =========================================================
# 2) 데이터 준비
# =========================================================
def prepare_tabular(df: pd.DataFrame, target_col: str):
    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}'이 train에 없습니다.")
    y_raw = df[target_col].to_numpy()
    classes, y = np.unique(y_raw, return_inverse=True)
    num_classes = len(classes)

    id_candidates = [c for c in df.columns if c.lower() in ("id", "index")]
    drop_cols = set(id_candidates + [target_col])
    feat_df = df.drop(columns=list(drop_cols), errors="ignore")

    cat_cols = [c for c in feat_df.columns if feat_df[c].dtype == "object"]
    num_cols = [c for c in feat_df.columns if c not in cat_cols]

    if len(num_cols) > 0:
        feat_df[num_cols] = feat_df[num_cols].apply(
            lambda s: pd.to_numeric(s, errors="coerce")
        ).fillna(0)

    cat_maps, cat_codes, cat_cardinalities = {}, [], []
    for c in cat_cols:
        cat_series = feat_df[c].astype("category")
        codes = cat_series.cat.codes.to_numpy().astype(np.int64)
        codes[codes < 0] = cat_series.cat.categories.size
        cat_codes.append(codes)
        cat_maps[c] = list(cat_series.cat.categories) + ["<NA>"]
        cat_cardinalities.append(len(cat_maps[c]))

    num_values, num_means, num_stds = [], {}, {}
    for c in num_cols:
        v = feat_df[c].to_numpy(dtype=np.float32)
        m = float(np.nanmean(v)) if np.isnan(v).any() else float(v.mean())
        s = float(np.nanstd(v)) if np.isnan(v).any() else float(v.std() + 1e-6)
        if s == 0.0: s = 1.0
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
        self.X_cat = torch.from_numpy(X_cat)
        self.X_num = torch.from_numpy(X_num)
        self.y = None if y is None else torch.from_numpy(y)

    def __len__(self):
        return self.X_cat.shape[0] if self.X_cat.numel() > 0 else self.X_num.shape[0]

    def __getitem__(self, idx):
        if self.y is None:
            return self.X_cat[idx], self.X_num[idx]
        return self.X_cat[idx], self.X_num[idx], self.y[idx]


# =========================================================
# 3) 모델 정의
# =========================================================
class FeatureTokenizer(nn.Module):
    def __init__(self, cat_cardinalities, num_feat_dim, d_model, token_dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.n_cat = len(cat_cardinalities)
        self.n_num = int(num_feat_dim)
        self.token_dropout = token_dropout

        self.cat_embeds = nn.ModuleList([nn.Embedding(c, d_model) for c in cat_cardinalities]) if self.n_cat > 0 else None
        self.num_proj = nn.ModuleList([nn.Linear(1, d_model) for _ in range(self.n_num)]) if self.n_num > 0 else None

        self.feature_pos = nn.Embedding(self.n_cat + self.n_num + 1, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, x_cat, x_num):
        B = x_cat.shape[0] if x_cat.numel() > 0 else x_num.shape[0]
        tokens = []
        if self.n_cat > 0:
            for i in range(self.n_cat):
                tokens.append(self.cat_embeds[i](x_cat[:, i]).unsqueeze(1))
        if self.n_num > 0:
            for j in range(self.n_num):
                tokens.append(self.num_proj[j](x_num[:, j:j+1]).unsqueeze(1))
        if len(tokens) == 0:
            raise RuntimeError("토큰 없음")

        x = torch.cat(tokens, dim=1)
        L = x.shape[1]

        if self.training and self.token_dropout > 0.0 and L > 1:
            mask = (torch.rand(B, L, device=x.device) < self.token_dropout).unsqueeze(-1)
            x = x.masked_fill(mask, 0.0)

        cls_tok = self.cls.expand(B, -1, -1)
        x = torch.cat([cls_tok, x], dim=1)
        pos_ids = torch.arange(L + 1, device=x.device).unsqueeze(0).expand(B, -1)
        x = x + self.feature_pos(pos_ids)
        return x


class TabTransformer(nn.Module):
    def __init__(self, cat_cardinalities, num_feat_dim, d_model=128,
                 n_heads=8, n_layers=4, ff_mult=4, dropout=0.2,
                 attn_dropout=0.0, layer_norm_eps=1e-5, num_classes=2,
                 token_dropout=0.0):
        super().__init__()
        self.tokenizer = FeatureTokenizer(cat_cardinalities, num_feat_dim, d_model, token_dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*ff_mult,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        for m in enc_layer.modules():
            if isinstance(m, nn.MultiheadAttention):
                m.dropout = attn_dropout
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.head = nn.Linear(d_model, num_classes if num_classes > 2 else 1)

    def forward(self, x_cat, x_num):
        x = self.tokenizer(x_cat, x_num)
        x = self.encoder(x)
        cls = self.norm(x[:, 0, :])
        return self.head(cls)


# =========================================================
# 4) 학습 유틸
# =========================================================
class FocalLoss(nn.Module):
    """
    Focal loss with optional per-class alpha vector.
    - gamma: focusing parameter
    - alpha: None | scalar | 1D tensor (num_classes)
    """
    def __init__(self, gamma: float = 2.0, alpha=None, reduction: str = "mean"):
        super().__init__()
        self.gamma = float(gamma)
        self.reduction = reduction

        if alpha is None:
            self.register_buffer("alpha", None)
        else:
            if isinstance(alpha, (list, np.ndarray)):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            elif isinstance(alpha, (int, float)):
                alpha = torch.tensor([float(alpha)], dtype=torch.float32)  # scalar
            assert alpha.ndim in (1,), "alpha must be 1D or scalar"
            self.register_buffer("alpha", alpha)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: [B, C], target: [B]
        logpt = F.log_softmax(logits, dim=1)                       # [B, C]
        pt = logpt.exp()                                           # [B, C]
        logpt = logpt.gather(1, target.view(-1, 1)).squeeze(1)     # [B]
        pt = pt.gather(1, target.view(-1, 1)).squeeze(1)           # [B]

        loss = -(1 - pt).pow(self.gamma) * logpt                   # [B]

        if self.alpha is not None:
            a = self.alpha
            if a.numel() == 1:
                loss = a.to(target.device) * loss
            else:
                a = a.to(target.device)                            # per-class
                loss = a[target] * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def build_optimizer(params, name, lr, weight_decay):
    name = name.lower()
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9, nesterov=True)
    raise ValueError(f"Unknown optimizer {name}")


def cosine_with_warmup(optimizer, warmup_steps, total_steps, min_lr=1e-6):
    def fn(step):
        if step < warmup_steps:
            return max(1e-8, step/max(1,warmup_steps))
        progress = (step-warmup_steps)/max(1,total_steps-warmup_steps)
        return max(min_lr, 0.5*(1+math.cos(math.pi*progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


def build_scheduler(optimizer, name, total_steps, warmup_ratio, min_lr, step_size, gamma):
    warmup_steps = int(warmup_ratio*total_steps)
    if name=="cosine":
        return cosine_with_warmup(optimizer, warmup_steps, total_steps, min_lr)
    if name=="step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    return None


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    y_true, y_pred = [], []
    for x_cat, x_num, y in loader:
        x_cat, x_num = x_cat.to(device), x_num.to(device)
        logits = model(x_cat, x_num)
        if num_classes==2:
            pred = (logits.squeeze(1) > 0).long().cpu().numpy()
        else:
            pred = logits.argmax(dim=1).cpu().numpy()
        y_pred.extend(pred.tolist())
        y_true.extend(y.numpy().tolist())
    f1 = f1_score(y_true, y_pred, average="macro") if len(set(y_true))>1 else 0.0
    acc = accuracy_score(y_true, y_pred) if len(y_true)>0 else 0.0
    return f1, acc, (y_true, y_pred)


def _flatten_metrics_for_csv(args, metrics):
    """
    args(하이퍼파라미터) + metrics(최종평가) -> 1행(dict)로 평탄화
    """
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        # ---- 기본 학습 설정
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "min_lr": args.min_lr,
        "weight_decay": args.weight_decay,
        "optimizer": args.optimizer,
        "sched": args.sched,
        "warmup_ratio": args.warmup_ratio,
        "step_size": args.step_size,
        "gamma": args.gamma,
        "label_smoothing": args.label_smoothing,
        "use_focal": bool(args.use_focal),
        "focal_gamma": args.focal_gamma,
        "focal_alpha": args.focal_alpha,
        "alpha_vec": getattr(args, "alpha_vec", ""),
        # ---- 모델 구조
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "ff_mult": args.ff_mult,
        "dropout": args.dropout,
        "attn_dropout": args.attn_dropout,
        "token_dropout": args.token_dropout,
        "layer_norm_eps": args.layer_norm_eps,
        # ---- 기타
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "grad_clip": args.grad_clip,
        "patience": args.patience,
        # ---- 총괄 성능
        "accuracy": metrics.get("accuracy", None),
        "precision_macro": metrics.get("precision_macro", None),
        "recall_macro": metrics.get("recall_macro", None),
        "f1_macro": metrics.get("f1_macro", None),
    }

    # per-class (라벨이 0,1,2 처럼 정수라고 가정; 문자열이면 그대로)
    per_class = metrics.get("per_class", {})
    for lab in sorted(per_class.keys(), key=lambda x: int(x) if str(x).isdigit() else x):
        pc = per_class[lab]
        row[f"class_{lab}_acc"] = pc.get("acc", None)
        row[f"class_{lab}_precision"] = pc.get("precision", None)
        row[f"class_{lab}_recall"] = pc.get("recall", None)
        row[f"class_{lab}_f1"] = pc.get("f1", None)
        row[f"class_{lab}_support"] = pc.get("support", None)
    return row


def _append_row_to_csv(row: dict, csv_path: str):
    """
    row(dict)를 csv에 1행 append (헤더는 파일 없을 때만 기록)
    """
    try:
        df = pd.DataFrame([row])
        header = not os.path.exists(csv_path)
        # Excel 친화적 인코딩
        df.to_csv(csv_path, mode="a", header=header, index=False, encoding="utf-8-sig")
    except Exception as e:
        # 조용히 무시(모델 학습 흐름 방해하지 않도록)
        pass


# =========================================================
# 5) 학습 루틴
# =========================================================

def train_and_eval(args, device):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # 기존 핸들러 전부 제거해서 중복 방지
    if logger.hasHandlers():
        logger.handlers.clear()

    # 콘솔 출력 핸들러 하나만 추가
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # 아티팩트 저장 모드일 경우, 파일 핸들러 추가
    if args.produce_artifacts:
        os.makedirs(args.save_dir, exist_ok=True)
        log_path = os.path.join(args.save_dir, "run.log")
        fh = logging.FileHandler(log_path, mode="w")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        logger.addHandler(fh)

    # ---------------- 데이터 준비 ----------------
    df_train = pd.read_csv(args.train_path)
    Xc, Xn, y, num_classes, meta = prepare_tabular(df_train, args.target)

    # 표준화
    for j, c in enumerate(meta["num_cols"]):
        m, s = meta["num_means"][c], meta["num_stds"][c]
        if s == 0: s = 1.0
        Xn[:, j] = (Xn[:, j] - m) / s

    stratify_vec = None if args.no_stratify or len(np.unique(y)) < 2 else y
    Xc_tr, Xc_val, Xn_tr, Xn_val, y_tr, y_val = train_test_split(
        Xc, Xn, y, test_size=args.val_ratio,
        random_state=args.seed, stratify=stratify_vec
    )
    # === 클래스별 alpha 벡터 자동 계산 (criterion 만들기 전에!) ===
    alpha_vec = None  # 기본값
    if args.use_focal and args.alpha_vec.strip():
        try:
            alpha_list = [float(x.strip()) for x in args.alpha_vec.split(",") if x.strip()]
        except Exception as e:
            raise ValueError(f"--alpha_vec 파싱 실패: {e}")
        if len(alpha_list) != num_classes:
            raise ValueError(f"--alpha_vec 길이({len(alpha_list)})가 클래스 수({num_classes})와 다릅니다.")
        alpha_vec = np.asarray(alpha_list, dtype=np.float32)
        # 안정성(음수/0 방지)
        alpha_vec = np.clip(alpha_vec, 1e-8, None)


    tr_loader = DataLoader(TabularDataset(Xc_tr, Xn_tr, y_tr),
                           batch_size=args.batch_size, shuffle=True)
    va_loader = DataLoader(TabularDataset(Xc_val, Xn_val, y_val),
                           batch_size=args.batch_size, shuffle=False)

    model = TabTransformer(
        meta["cat_cardinalities"], len(meta["num_cols"]),
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        ff_mult=args.ff_mult, dropout=args.dropout,
        attn_dropout=args.attn_dropout, layer_norm_eps=args.layer_norm_eps,
        num_classes=num_classes, token_dropout=args.token_dropout
    ).to(device)

    optimizer = build_optimizer(model.parameters(), args.optimizer,
                                args.lr, args.weight_decay)
    total_steps = args.epochs * max(1, len(tr_loader))
    scheduler = build_scheduler(optimizer, args.sched, total_steps,
                                args.warmup_ratio, args.min_lr,
                                args.step_size, args.gamma)

    # === 손실함수 생성 ===
    if args.use_focal:
        # alpha_vec이 있으면 per-class, 없으면 스칼라 focal_alpha 사용
        focal_alpha = alpha_vec if (alpha_vec is not None) else args.focal_alpha
        _focal = FocalLoss(gamma=args.focal_gamma, alpha=focal_alpha, reduction="mean")

        def criterion(logits, y, num_classes):
            return _focal(logits, y)
    else:
        def criterion(logits, y, num_classes):
            return F.cross_entropy(logits, y, label_smoothing=args.label_smoothing)

    # ---------------- 학습 루프 ----------------
    best_f1, best_state, no_improve = -1.0, None, 0

    # tqdm 한 줄 + 타임스탬프 postfix (훈련 중에는 이것만 보임)
    with tqdm(total=args.epochs, desc="Training Epochs", leave=True) as pbar:
        pbar.set_postfix_str(datetime.now().strftime("[%Y-%m-%d %H:%M:%S]"))

        for epoch in range(1, args.epochs + 1):
            model.train()
            running = 0.0

            for x_cat, x_num, yb in tr_loader:
                x_cat, x_num, yb = x_cat.to(device), x_num.to(device), yb.to(device)
                logits = model(x_cat, x_num)

                if criterion:
                    loss = criterion(logits, yb, num_classes)
                else:
                    if num_classes == 2:
                        loss = F.binary_cross_entropy_with_logits(
                            logits.squeeze(1), yb.float()
                        )
                    else:
                        loss = F.cross_entropy(
                            logits, yb, label_smoothing=args.label_smoothing
                        )

                optimizer.zero_grad()
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                if scheduler and args.sched != "step":
                    scheduler.step()
                running += loss.item()

            if scheduler and args.sched == "step":
                scheduler.step()

            # 검증 (에폭 중간 로그는 전혀 출력하지 않음)
            f1, acc, _ = evaluate(model, va_loader, device, num_classes)

            # 베스트 갱신 & 얼리 스탑 카운터
            if f1 > best_f1:
                best_f1 = f1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            # 진행 바는 한 줄만, 타임스탬프만 갱신
            pbar.update(1)
            pbar.set_postfix_str(datetime.now().strftime("[%Y-%m-%d %H:%M:%S]"))

            if args.patience > 0 and no_improve >= args.patience:
                # 얼리 스탑 메시지도 훈련 중엔 출력하지 않음 (조용히 break)
                break

    # ---------------- 최종 평가 ----------------
    # 베스트 가중치로 로드
    if best_state:
        model.load_state_dict(best_state)

    # 최종 검증
    f1, acc, (yt, yp) = evaluate(model, va_loader, device, num_classes)

    # 클래스/혼동행렬/클래스별 acc
    labels = sorted(np.unique(yt))
    cm = confusion_matrix(yt, yp, labels=labels)
    support = cm.sum(axis=1)
    acc_per_class = (np.diag(cm) / np.maximum(support, 1)).tolist()

    # 클래스별 precision/recall/f1, 전체 macro precision/recall/f1
    prec_pc, rec_pc, f1_pc, sup_pc = precision_recall_fscore_support(
        yt, yp, labels=labels, average=None, zero_division=0
    )
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        yt, yp, average="macro", zero_division=0
    )

    # 공통으로 쓸 metrics 딕셔너리 생성 (← 이게 없어서 NameError였음)
    metrics = {
        "accuracy": float(acc),
        "precision_macro": float(macro_prec),
        "recall_macro": float(macro_rec),
        "f1_macro": float(macro_f1),
        "per_class": {
            str(lab): {
                "acc": float(acc_i),
                "precision": float(p_i),
                "recall": float(r_i),
                "f1": float(f1_i),
                "support": int(s)
            }
            for lab, acc_i, p_i, r_i, f1_i, s in zip(labels, acc_per_class, prec_pc, rec_pc, f1_pc, sup_pc)
        }
    }

    # ===== 최종 출력 (훈련 끝난 뒤에만) =====
    print("===== 최종 평가 결과 =====")
    for lab in labels:
        pc = metrics["per_class"][str(lab)]
        print(f"{lab} : acc {pc['acc']:.4f}, precision {pc['precision']:.4f}, "
              f"recall {pc['recall']:.4f}, f1 {pc['f1']:.4f}")
    print(f"전체 accuracy : {metrics['accuracy']:.4f}")
    print(f"precision_macro : {metrics['precision_macro']:.4f}")
    print(f"recall_macro : {metrics['recall_macro']:.4f}")
    print(f"f1_macro : {metrics['f1_macro']:.4f}")

    # 저장 (옵션)
    if args.produce_artifacts:
        os.makedirs(args.save_dir, exist_ok=True)
        with open(os.path.join(args.save_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)
        with open(os.path.join(args.save_dir, 'params.json'), 'w') as f:
            json.dump(vars(args), f, indent=4, ensure_ascii=False)
        _save_confusion_matrix(yt, yp, args.save_dir)

    # ===== CSV 로그 (선택) =====
    if LOG_TO_CSV:
        row = _flatten_metrics_for_csv(args, metrics)
        _append_row_to_csv(row, CSV_LOG_PATH)

    return {"model": model, "metrics": metrics, "params": vars(args)}


def _save_confusion_matrix(y_true, y_pred, save_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    plt.close()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 기존 logging.info → logger.info
    logger = logging.getLogger(__name__)
    logger.info(f"Using device: {device}")

    train_and_eval(args, device)


if __name__ == "__main__":
    main()