#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ft_transformer_classifier.py (multiclass for support_needs)
- [수정] 최신 PyTorch 호환을 위해 pytorch-frame 라이브러리 사용
"""

import os
import json
import argparse
import logging
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score, precision_recall_fscore_support, accuracy_score, confusion_matrix
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# [수정] pytorch-frame 라이브러리 import
import torch_frame
from torch_frame.data import Dataset
from torch_frame.nn import FTTransformer

# ─────────────────────────────────────────────────────────────────────
# 기본 하이퍼파라미터 (FT-Transformer 용)
# ─────────────────────────────────────────────────────────────────────
DEFAULT_PARAMS = dict(
    # --- 모델 구조 (Architecture) ---
    channels=192,  # d_token과 유사한 개념
    num_layers=3,  # n_blocks와 유사
    num_heads=8,  # 어텐션 헤드 개수
    attn_dropout=0.2,
    ffn_dropout=0.1,

    # --- 학습 (Training) ---
    learning_rate=1e-4,
    weight_decay=1e-5,
    batch_size=256,
    n_epochs=100,

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
    # --- 1. 파라미터 및 준비 ---
    params = DEFAULT_PARAMS.copy()
    if params_dict:
        params.update(params_dict)

    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    logger = logging.getLogger(__name__)
    # ... (로거 설정은 이전과 동일) ...

    # --- 2. [수정] pytorch-frame을 사용한 데이터 준비 ---
    logger.info("데이터를 불러오고 pytorch-frame Dataset을 생성합니다...")
    df = pd.read_csv(train_path)

    # ID 제거 및 타겟 타입 변경
    drop_cols = [c for c in ["ID", "id", "Id"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    df[target_col] = df[target_col].astype(int)

    # 데이터 분할
    train_df, val_df = train_test_split(
        df, test_size=valid_size, random_state=seed, stratify=df[target_col]
    )

    # 피처의 의미(semantic type)를 정의
    col_to_stype = {
        col: torch_frame.categorical for col in df.columns if df[col].dtype == 'object'
    }
    col_to_stype.update({
        col: torch_frame.numerical for col in df.columns if col not in col_to_stype and col != target_col
    })

    # pytorch-frame Dataset 생성
    dataset = Dataset(
        df=train_df,
        col_to_stype=col_to_stype,
        target_col=target_col
    )

    # 데이터셋의 통계 정보를 계산하고 텐서로 변환 (내부적으로 스케일링, 인코딩 수행)
    dataset.materialize()

    # 훈련 및 검증용 DataLoader 생성
    train_dataset = dataset
    val_dataset = dataset.copy(val_df)

    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'])

    # --- 3. 모델, 손실함수, 옵티마이저 정의 ---
    model = FTTransformer(
        channels=params['channels'],
        out_channels=dataset.num_classes,
        num_layers=params['num_layers'],
        num_heads=params['num_heads'],
        attn_dropout=params['attn_dropout'],
        ffn_dropout=params['ffn_dropout'],
        col_stats=dataset.col_stats,
        col_names_dict=dataset.col_names_dict,
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])

    # --- 4. 학습 루프 ---
    logger.info("모델 학습을 시작합니다...")
    best_loss = float('inf')
    patience_counter = 0
    patience = 10

    for epoch in range(params['n_epochs']):
        model.train()
        for tf in train_loader:
            tf = tf.to(device)
            optimizer.zero_grad()
            outputs = model(tf)
            loss = loss_fn(outputs, tf.y)
            loss.backward()
            optimizer.step()

        # 검증
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for tf in val_loader:
                tf = tf.to(device)
                outputs = model(tf)
                val_loss += loss_fn(outputs, tf.y).item()
        val_loss /= len(val_loader)

        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch + 1}/{params['n_epochs']}, Validation Loss: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            if produce_artifacts:
                torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break
    logger.info("모델 학습 완료.")

    # --- 5. 최종 평가 ---
    logger.info("모델 평가를 시작합니다...")
    if produce_artifacts:
        model.load_state_dict(torch.load(os.path.join(save_dir, "model.pt")))

    model.eval()
    all_preds = []
    all_true = []
    with torch.no_grad():
        for tf in val_loader:
            tf = tf.to(device)
            outputs = model(tf)
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu())
            all_true.append(tf.y.cpu())

    preds_np = torch.cat(all_preds).numpy()
    true_np = torch.cat(all_true).numpy()
    metrics = _calculate_metrics(true_np, preds_np)

    # ... (이하 결과 저장, 제출 파일 생성, 최종 평가 결과 출력 로직은 이전과 동일) ...

    return dict(model=model, metrics=metrics, params=params)

# ... (이하 _calculate_metrics, _save_confusion_matrix, main 함수는 이전과 동일) ...
# main 함수에서는 save_dir의 기본값을 "results/ft_transformer"로 변경하는 것을 추천