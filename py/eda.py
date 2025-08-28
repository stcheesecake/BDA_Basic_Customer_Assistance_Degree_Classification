
import argparse
import math
import pandas as pd
import numpy as np
from typing import Optional

# --------- helpers ---------

def sample_value_type(series: pd.Series, sample_size: int = 500) -> Optional[str]:
    """Non-NA 중 최대 sample_size개 샘플링해서 실제 파이썬 타입의 최빈값을 반환.
    전부 NA면 None."""
    s = series.dropna()
    if s.empty:
        return None
    if len(s) > sample_size:
        s = s.sample(sample_size, random_state=42)
    # object 컬럼에서 numpy 타입이 섞일 수 있어 type 명만 문자열로
    types = s.map(lambda x: type(x).__name__)
    return types.mode().iat[0] if not types.empty else None

def make_flags(
    pandas_dtype: str,
    non_missing: int,
    nunique: int,
    total: int,
    long_text_threshold: int = 100,    # "문장형" 추정 기준: 고유값이 충분히 많음
    constant_ratio_hi: float = 0.99,   # 거의 상수
) -> str:
    flags = []

    # 잠재적 ID: 결측 제외 고유값이 거의 모두 다른 값
    if non_missing > 0 and nunique == non_missing:
        flags.append("maybe_id_or_free_text")

    # 거의 상수(정보량 낮음)
    if non_missing > 0:
        unique_ratio = nunique / float(non_missing)
        # 상위 빈도 비율이 0.99 이상이면 더 확실하지만,
        # 여기서는 빠른 휴리스틱으로 unique_ratio가 매우 낮은 것도 참고
        if unique_ratio < (1 - constant_ratio_hi):
            flags.append("low_variation")

    # 문장형 추정: object이고 고유값이 충분히 많음
    if pandas_dtype == "object" and nunique >= long_text_threshold:
        flags.append("maybe_long_text")

    return ",".join(flags) if flags else ""

# --------- main ---------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", default = "data/train.csv", help="Path to train.csv")
    parser.add_argument("--output", "-o", default="results/eda/feature_eda.csv", help="Path to save EDA CSV")
    parser.add_argument("--long_text_threshold", type=int, default=100,
                        help="고유값 개수가 이 값 이상이면 문장형 추정")
    parser.add_argument("--constant_ratio_hi", type=float, default=0.99,
                        help="상위 단일값 비율이 이 값 이상이면 거의 상수로 간주 (휴리스틱)")
    parser.add_argument("--sample_type_size", type=int, default=500,
                        help="value_type_example 추정을 위한 샘플 크기")
    args = parser.parse_args()

    # CSV 로드 (인코딩 이슈가 있으면 encoding 인자를 지정하세요: e.g., encoding='utf-8-sig')
    df = pd.read_csv(args.input)

    total_samples = len(df)
    results = []

    for col in df.columns:
        series = df[col]
        non_missing = int(series.notna().sum())
        nunique = int(series.nunique(dropna=True))
        pandas_dtype = str(series.dtype)
        value_type = sample_value_type(series, sample_size=args.sample_type_size)
        unique_ratio = float(nunique / non_missing) if non_missing > 0 else np.nan

        # 플래그(휴리스틱): 잠재적 ID / 문장형 / 저정보량
        flags = make_flags(
            pandas_dtype=pandas_dtype,
            non_missing=non_missing,
            nunique=nunique,
            total=total_samples,
            long_text_threshold=args.long_text_threshold,
            constant_ratio_hi=args.constant_ratio_hi
        )

        results.append({
            "feature": col,
            "total_samples": total_samples,
            "non_missing_samples": non_missing,
            "pandas_dtype": pandas_dtype,
            "value_type_example": value_type if value_type is not None else "",
            "unique_values": nunique,
            "unique_ratio": round(unique_ratio, 6) if not math.isnan(unique_ratio) else "",
            "flags": flags
        })

    eda_df = pd.DataFrame(results).sort_values(["pandas_dtype", "feature"]).reset_index(drop=True)

    # 콘솔 출력 (전체 보기)
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 180):
        print("\n=== Feature EDA Summary ===")
        print(eda_df)

    # CSV 저장
    eda_df.to_csv(args.output, index=False)
    print(f"\nSaved EDA CSV -> {args.output}")

if __name__ == "__main__":
    main()