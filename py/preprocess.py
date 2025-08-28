#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train/Test OOF 변환 파이프라인 (+ 문제 피처 자동 제외)

- 입력:
  --train_input   (기본: data/train.csv)
  --test_input    (기본: data/test.csv)
- 동작:
  1) train: 문자열(object) & nunique > unique_count 인 칼럼들만 OOF 확률(1D) 생성
     - (사전 필터) 품질 낮은 텍스트 피처는 OOF 후보에서 제외
     - 누출 방지: 벡터라이저/모델 모두 fold 내 학습 폴드에서만 fit
     - (사후 필터) fold의 절반 이상이 prior 폴백이면 최종 제외
     - 생성된 *_oof_prob 붙이고, 원본 텍스트 칼럼 제거
  2) test : 위와 동일한 칼럼들에 대해 train 전체로 TF-IDF/LogReg full-fit → test 확률(1D) 생성
     - *_oof_prob 붙이고, 원본 텍스트 칼럼 제거
  3) excepted_feature에 지정된 컬럼은 train/test 최종 CSV에서 완전히 제거
  4) 최종 저장 직전: 남은 범주형(object/category)만 NaN→"__MISSING__" 처리 + 문자열 캐스팅
- 출력:
  --train_output  (기본: data/preprocessed_train_oof.csv)
  --test_output   (기본: data/preprocessed_test_oof.csv)
"""

import os
import argparse
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# --------------------- Args ---------------------

def parse_args():
    parser = argparse.ArgumentParser()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")

    parser.add_argument("--train_input", default=os.path.join(data_dir, "train.csv"),
                        help="학습 CSV 경로 (기본: data/train.csv)")
    parser.add_argument("--test_input", default=os.path.join(data_dir, "test.csv"),
                        help="테스트 CSV 경로 (기본: data/test.csv)")

    parser.add_argument("--train_output", default=os.path.join(data_dir, "preprocessed_train_oof.csv"),
                        help="변환된 학습 CSV 출력 경로 (기본: data/preprocessed_train_oof.csv)")
    parser.add_argument("--test_output", default=os.path.join(data_dir, "preprocessed_test_oof.csv"),
                        help="변환된 테스트 CSV 출력 경로 (기본: data/preprocessed_test_oof.csv)")

    parser.add_argument("--unique_count", type=int, default=100,
                        help="이 값보다 unique 수가 많은 문자열 feature만 OOF 1D로 변환")
    parser.add_argument("--excepted_feature", default="ID,class3,class4,contest_participitation,idea_contest,contest_award, completed_semester",
                        help='최종 CSV에서 완전히 제외할 컬럼명 리스트(콤마 구분). 예: "ID,class3,class4"')

    parser.add_argument("--target", default="withdrawal",
                        help="타깃 컬럼명 (train에 존재해야 함; 기본: withdrawal)")
    parser.add_argument("--n_splits", type=int, default=5,
                        help="OOF K-Fold 개수 (기본: 5)")
    parser.add_argument("--random_state", type=int, default=42,
                        help="재현성 시드")
    parser.add_argument("--encoding", default=None,
                        help="CSV 인코딩(필요시 지정. 예: utf-8-sig)")

    # 사전/사후 필터 기준 (필요시 조정)
    parser.add_argument("--min_non_empty_ratio", type=float, default=0.30,
                        help="사전 필터: 비어있지 않은 비율 최소치(기본 0.30)")
    parser.add_argument("--min_median_len", type=int, default=3,
                        help="사전 필터: 텍스트 중앙 길이 최소치(기본 3)")
    parser.add_argument("--min_non_missing", type=int, default=200,
                        help="사전 필터: 결측 제외 유효 샘플 최소치(기본 200)")
    parser.add_argument("--max_prior_folds_ratio", type=float, default=0.5,
                        help="사후 필터: prior 폴백 fold 비율 임계(기본 0.5 이상이면 제외)")

    return parser.parse_args()


def to_list_from_csv_arg(arg: str) -> List[str]:
    arg = arg.strip()
    if not arg:
        return []
    return [x.strip() for x in arg.split(",") if x.strip()]


# ---------- 안전한 TF-IDF 폴백 ----------

def fit_tfidf_safe(train_texts: pd.Series,
                   val_texts: Optional[pd.Series] = None,
                   min_df_try: int = 3,
                   max_features: int = 100_000):
    """
    - Word TF-IDF 시도 (min_df를 학습 폴드 크기에 맞춰 자동 다운스케일)
    - 실패 시 char_wb TF-IDF 폴백
    - 둘 다 실패하면 (전부 공백/희소) → (vec, None, None) 반환
    """
    train_texts = train_texts.fillna("").astype(str).str.strip()
    if val_texts is not None:
        val_texts = val_texts.fillna("").astype(str).str.strip()

    n_docs_nonempty = int((train_texts.str.len() > 0).sum())
    min_df_eff = min_df_try if n_docs_nonempty >= min_df_try else 1

    # 1) word 기반
    try:
        vec = TfidfVectorizer(min_df=min_df_eff, ngram_range=(1, 2), max_features=max_features)
        Xtr = vec.fit_transform(train_texts)
        if Xtr.shape[1] > 0:
            Xva = vec.transform(val_texts) if val_texts is not None else None
            return vec, Xtr, Xva
    except Exception:
        pass

    # 2) char_wb 기반 (한국어/짧은 텍스트에 강함)
    try:
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                              min_df=min_df_eff, max_features=max_features)
        Xtr = vec.fit_transform(train_texts)
        if Xtr.shape[1] > 0:
            Xva = vec.transform(val_texts) if val_texts is not None else None
            return vec, Xtr, Xva
    except Exception:
        pass

    return None, None, None


# ---------- OOF/Full-fit 생성기 ----------

def make_oof_feature(texts: pd.Series,
                     y: np.ndarray,
                     n_splits: int = 5,
                     random_state: int = 42) -> Tuple[np.ndarray, int]:
    """
    문자열 Series -> OOF 예측확률(1D) 생성
    - TF-IDF + LogisticRegression(class_weight='balanced')
    - 누출 방지: 벡터라이저/모델 모두 fold 내 학습 폴드로만 fit
    - 안전장치: 토큰이 0개면 prior 확률(해당 fold의 y_tr 평균)로 대체
    반환: (oof_pred, prior_folds_count)
    """
    texts = texts.fillna("").astype(str)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = np.zeros(len(texts), dtype=np.float32)
    prior_folds = 0

    for tr_idx, va_idx in skf.split(np.zeros(len(y)), y):
        X_tr_text = texts.iloc[tr_idx]
        y_tr = y[tr_idx]
        X_va_text = texts.iloc[va_idx]

        vec, Xtr, Xva = fit_tfidf_safe(X_tr_text, X_va_text, min_df_try=3, max_features=100_000)
        if vec is None or Xtr is None or Xtr.shape[1] == 0:
            prior = float(np.mean(y_tr))
            oof[va_idx] = prior
            prior_folds += 1
            print("[경고] 해당 fold에서 유효 토큰이 없어 prior 확률로 대체:", prior)
            continue

        clf = LogisticRegression(
            solver="liblinear",
            C=1.0,
            class_weight="balanced",
            max_iter=300
        )
        clf.fit(Xtr, y_tr)
        oof[va_idx] = clf.predict_proba(Xva)[:, 1].astype(np.float32)

    return oof, prior_folds


def fit_full_and_predict_prob(texts_train: pd.Series,
                              y_train: np.ndarray,
                              texts_test: pd.Series) -> np.ndarray:
    """
    train 전체로 TF-IDF/LogReg full-fit → test 확률(1D) 생성
    - 안전장치: 토큰이 0개면 prior 확률로 채움
    """
    texts_train = texts_train.fillna("").astype(str)
    texts_test = texts_test.fillna("").astype(str)

    vec, Xtr, Xte = fit_tfidf_safe(texts_train, texts_test, min_df_try=3, max_features=100_000)
    if vec is None or Xtr is None or Xtr.shape[1] == 0:
        prior = float(np.mean(y_train))
        print("[경고] full-fit에서도 유효 토큰이 없어 prior 확률로 대체:", prior)
        return np.full(len(texts_test), prior, dtype=np.float32)

    clf = LogisticRegression(
        solver="liblinear",
        C=1.0,
        class_weight="balanced",
        max_iter=300
    )
    clf.fit(Xtr, y_train)
    prob = clf.predict_proba(Xte)[:, 1].astype(np.float32)
    return prob


# ---------- 범주형 정규화(결측/타입) ----------

def normalize_categorical(df: pd.DataFrame,
                          exclude_cols: List[str],
                          oof_suffix: str = "_oof_prob") -> pd.DataFrame:
    """
    - 남아 있는 범주형(object/category) 컬럼만 대상으로 NaN→'__MISSING__' 후 문자열 캐스팅
    - OOF 컬럼( *_oof_prob )과 제외목록은 건드리지 않음
    """
    obj_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    obj_cols = [c for c in obj_cols if c not in exclude_cols and not c.endswith(oof_suffix)]
    if not obj_cols:
        return df
    df[obj_cols] = df[obj_cols].apply(lambda s: s.fillna("__MISSING__")).astype(str)
    return df


# ---------- 사전/사후 필터 ----------

def is_degenerate_text_feature(s: pd.Series,
                               non_empty_ratio_min: float = 0.30,
                               median_len_min: int = 3,
                               min_non_missing: int = 200) -> bool:
    """
    텍스트 품질/규모가 낮아 보이면 True (OOF 후보에서 제외)
    - ID-유사: nunique == non_missing
    - 결측/빈문자 과다: non_empty_ratio < 임계
    - 텍스트 너무 짧음: median_len < 임계
    - 유효 샘플 적음: non_missing < 임계
    """
    s0 = s.copy()
    non_missing = int(s0.notna().sum())
    nunique = int(s0.nunique(dropna=True))

    # ID-유사
    if non_missing > 0 and nunique == non_missing:
        return True

    ss = s0.fillna("").astype(str).str.strip()
    total = len(ss)
    non_empty = int((ss.str.len() > 0).sum())
    non_empty_ratio = non_empty / total if total > 0 else 0.0
    median_len = int(ss.str.len().median()) if total > 0 else 0

    if (non_empty_ratio < non_empty_ratio_min) or (median_len < median_len_min) or (non_missing < min_non_missing):
        return True
    return False


# --------------------- Main ---------------------

def main():
    args = parse_args()

    # 경로/출력 폴더 준비
    for p in [args.train_input, args.test_input]:
        if not os.path.exists(p):
            print(f"[경고] 파일이 없습니다: {p}")
    os.makedirs(os.path.dirname(args.train_output), exist_ok=True)
    os.makedirs(os.path.dirname(args.test_output), exist_ok=True)

    # 로드
    read_csv = (lambda path: pd.read_csv(path, encoding=args.encoding)) if args.encoding \
        else (lambda path: pd.read_csv(path))

    train_df = read_csv(args.train_input)
    if args.target not in train_df.columns:
        raise KeyError(f"타깃 컬럼 '{args.target}'이(가) train에 없습니다: {args.train_input}")

    test_exists = os.path.exists(args.test_input)
    test_df = read_csv(args.test_input) if test_exists else pd.DataFrame()

    # 제외 컬럼 파싱
    excepted = set(to_list_from_csv_arg(args.excepted_feature))
    missing_ex = [c for c in excepted if c not in train_df.columns and (not test_exists or c not in test_df.columns)]
    if missing_ex:
        print(f"[정보] excepted_feature에 있지만 파일에 없는 컬럼(무시): {missing_ex}")

    # 타깃 분리
    y = train_df[args.target].values
    X_train = train_df.drop(columns=[args.target])

    # test에 타깃이 있으면 제거
    X_test = test_df.drop(columns=[args.target], errors="ignore") if test_exists else pd.DataFrame()

    # OOF 대상 1차 선정
    candidates = []
    for col in X_train.columns:
        if col in excepted:
            continue
        if X_train[col].dtype == "object":
            nunique = X_train[col].nunique(dropna=True)
            if nunique > args.unique_count:
                candidates.append(col)
    print(f"[정보] 1차 OOF 후보({len(candidates)}개): {candidates}")

    # ---- 사전 필터: 품질 낮은 텍스트 제외 ----
    filtered = []
    for col in candidates:
        if is_degenerate_text_feature(
            X_train[col],
            non_empty_ratio_min=args.min_non_empty_ratio,
            median_len_min=args.min_median_len,
            min_non_missing=args.min_non_missing
        ):
            print(f"[제외/사전] '{col}' : 텍스트 품질/규모 낮음 → OOF 제외")
        else:
            filtered.append(col)
    candidates = filtered
    print(f"[정보] 사전 필터 후 OOF 후보({len(candidates)}개): {candidates}")

    # ---- Train: OOF 생성 ----
    new_train_cols: Dict[str, np.ndarray] = {}
    final_candidates: List[str] = []
    for col in candidates:
        print(f"[진행/Train-OOF] {col}")
        oof, prior_folds = make_oof_feature(
            texts=X_train[col],
            y=y,
            n_splits=args.n_splits,
            random_state=args.random_state,
        )
        # 사후 필터: prior 폴백 비율 체크
        if args.n_splits > 0 and (prior_folds / args.n_splits) > args.max_prior_folds_ratio:
            print(f"[제외/사후] '{col}' : prior 폴백 {prior_folds}/{args.n_splits} folds → 최종 제외")
            continue
        new_train_cols[f"{col}_oof_prob"] = oof
        final_candidates.append(col)

    candidates = final_candidates
    print(f"[정보] 최종 OOF 대상({len(candidates)}개): {candidates}")

    # 원본 텍스트 & 제외목록 제거 → 새 1D 컬럼 추가
    drop_train_cols = candidates + list(excepted)
    X_train_conv = X_train.drop(columns=drop_train_cols, errors="ignore")
    for new_name, arr in new_train_cols.items():
        X_train_conv[new_name] = arr
    out_train = pd.concat([X_train_conv, pd.Series(y, name=args.target)], axis=1)

    # ---- Test: full-fit → 확률 생성 ----
    new_test_cols: Dict[str, np.ndarray] = {}
    if test_exists and not X_test.empty and len(candidates) > 0:
        for col in candidates:
            if col not in X_test.columns:
                print(f"[경고] test에 {col} 이(가) 없어 변환 건너뜀")
                continue
            print(f"[진행/Test-FullFit] {col}")
            prob = fit_full_and_predict_prob(
                texts_train=X_train[col],
                y_train=y,
                texts_test=X_test[col]
            )
            new_test_cols[f"{col}_oof_prob"] = prob

        drop_test_cols = candidates + list(excepted)
        X_test_conv = X_test.drop(columns=drop_test_cols, errors="ignore")
        for new_name, arr in new_test_cols.items():
            X_test_conv[new_name] = arr
        out_test = X_test_conv.copy()
    else:
        out_test = X_test.copy()  # 빈 DF일 수 있음

    # ---- excepted_feature는 최종에서도 완전히 제거(이중확인) ----
    out_train = out_train.drop(columns=[c for c in excepted if c in out_train.columns], errors="ignore")
    out_test = out_test.drop(columns=[c for c in excepted if c in out_test.columns], errors="ignore")

    # ---- (중요) 범주형 정규화: NaN → "__MISSING__", 문자열 캐스팅 ----
    exclude_for_norm = list(excepted) + [args.target]
    out_train = normalize_categorical(out_train, exclude_cols=exclude_for_norm, oof_suffix="_oof_prob")
    if test_exists and not out_test.empty:
        out_test = normalize_categorical(out_test, exclude_cols=list(excepted), oof_suffix="_oof_prob")

    # ---- 저장 ----
    out_train.to_csv(args.train_output, index=False)
    print(f"[완료] Train 저장: {args.train_output}")

    if test_exists:
        out_test.to_csv(args.test_output, index=False)
        print(f"[완료] Test 저장:  {args.test_output}")
    else:
        print("[정보] test_input 파일이 없어 Test 변환을 건너뜀")

    # ---- 요약 ----
    print("\n[요약]")
    print(f"- train_input: {args.train_input}")
    print(f"- test_input : {args.test_input} ({'존재' if test_exists else '없음'})")
    print(f"- train_output: {args.train_output}")
    print(f"- test_output : {args.test_output if test_exists else '(생성안함)'}")
    print(f"- target: {args.target}")
    print(f"- unique_count 임계값: {args.unique_count}")
    print(f"- 제외된 컬럼: {sorted(list(excepted))}")
    print(f"- OOF 1차 후보 수: {len(filtered) + (len(candidates) - len(filtered)) if 'filtered' in locals() else 'N/A'}")
    print(f"- 최종 OOF 대상 수: {len(candidates)}")
    if candidates:
        print(f"- 제거된 원본 텍스트 컬럼(train): {candidates}")
        print(f"- 생성된 OOF 1D 컬럼(train): {list(new_train_cols.keys())}")
        if test_exists:
            kept_for_test = [k for k in candidates if k in X_test.columns]
            print(f"- test 변환 적용된 원본 텍스트 컬럼: {kept_for_test}")
            print(f"- 생성된 OOF 1D 컬럼(test): {list(new_test_cols.keys())}")
        else:
            print("- test 없음: test 변환 생략")


if __name__ == "__main__":
    main()
