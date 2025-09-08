# eda.py
import argparse
import os
import glob
import platform
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")   # 화면 출력 대신 파일 저장
import matplotlib.pyplot as plt
import seaborn as sns

MODEL_NAME = "xgboost"   # lightgbm, catboost, xgboost


# ──────────────────────────────
# 한글 폰트 설정
# ──────────────────────────────
def set_korean_font():
    system = platform.system()
    if system == "Windows":
        plt.rcParams["font.family"] = "Malgun Gothic"
    elif system == "Darwin":
        plt.rcParams["font.family"] = "AppleGothic"
    else:
        plt.rcParams["font.family"] = "NanumGothic"
    plt.rcParams["axes.unicode_minus"] = False

# ──────────────────────────────
# 유틸
# ──────────────────────────────
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def find_latest_hpo_csv(path_or_dir: str) -> str:
    # 파일이면 그대로, 디렉토리면 *_hpo.csv 중 최신 선택
    if os.path.isfile(path_or_dir):
        return path_or_dir
    pattern = os.path.join(path_or_dir, "*_hpo.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"HPO csv not found under: {path_or_dir}")
    return max(files, key=os.path.getmtime)

def detect_param_cols(df: pd.DataFrame):
    # hpo.csv 형식: trial, param_*, f1_macro, accuracy
    cand = [c for c in df.columns if c.startswith("param_")]
    if cand:
        return cand
    # 백업: 하위호환 (그냥 하이퍼파라미터 이름만 있을 수도 있음)
    skip = {"trial", "f1_macro", "accuracy"}
    return [c for c in df.columns if c not in skip]

def is_int_series(s: pd.Series) -> bool:
    if pd.api.types.is_integer_dtype(s):
        return True
    # float인데 값이 전부 정수 형태인 경우
    try:
        return np.allclose(s.dropna().values, np.round(s.dropna().values))
    except Exception:
        return False

# ──────────────────────────────
# HPO 분석 & 추천 범위 산출
# ──────────────────────────────
def analyze_hpo(hpo_csv: str, out_dir: str, top_frac: float = 0.2):
    ensure_dir(out_dir)
    df = pd.read_csv(hpo_csv)

    # 컬럼 확인
    if "f1_macro" not in df.columns:
        raise ValueError("hpo.csv에 f1_macro 컬럼이 없습니다.")
    if "accuracy" not in df.columns:
        print("⚠️ accuracy 컬럼이 없어도 계속 진행합니다.")
    param_cols = detect_param_cols(df)

    # 상위 top_frac 추출
    thr = df["f1_macro"].quantile(1 - top_frac)
    df["is_top"] = (df["f1_macro"] >= thr).astype(int)
    top = df[df["is_top"] == 1].copy()

    # ── 1) 파라미터 분포(전체 vs 상위) & 성능 ────────────────────
    n_params = len(param_cols)
    n_cols = 3
    n_rows = int(np.ceil(n_params / n_cols)) if n_params else 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4.2*n_rows), dpi=120)
    if n_params == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, col in enumerate(param_cols):
        ax = axes[i]
        # 전체 분포 + 상위 분포를 겹쳐 보여줌
        # (hue를 사용해 seaborn 0.14 palette 경고 방지)
        tmp = df[[col, "is_top"]].copy()
        tmp["is_top"] = tmp["is_top"].map({0: "all", 1: f"top {int(top_frac*100)}%"})

        # 연속/이산에 따라 다른 플롯
        if is_int_series(df[col]):
            # 이산형: countplot
            sns.countplot(data=tmp, x=col, hue="is_top", ax=ax)
            ax.set_xlabel(col)
            ax.set_ylabel("count")
        else:
            # 연속형: kde/hist
            sns.histplot(data=tmp, x=col, hue="is_top", element="step",
                         stat="density", common_norm=False, ax=ax)
            ax.set_xlabel(col)
            ax.set_ylabel("density")

        ax.set_title(f"{col} distribution (all vs top)")
        handles, labels = ax.get_legend_handles_labels()
        if any(lbl and not str(lbl).startswith("_") for lbl in labels):
            ax.legend(title="", loc="best")
        else:
            # legend 객체가 이미 있으면 제거
            if getattr(ax, "legend_", None) is not None:
                ax.legend_.remove()

    # 남는 축 제거
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    dist_path = os.path.join(out_dir, "hpo_param_distributions.png")
    plt.savefig(dist_path, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved -> {dist_path}")

    # ── 2) f1_macro vs 각 파라미터 관계(스캐터/라인) ───────────────
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4.2*n_rows), dpi=120)
    if n_params == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, col in enumerate(param_cols):
        ax = axes[i]
        if is_int_series(df[col]) and df[col].nunique() <= 30:
            # 이산이면 값별 평균 f1을 선으로
            m = df.groupby(col)["f1_macro"].mean().reset_index()
            sns.lineplot(data=m, x=col, y="f1_macro", marker="o", ax=ax)
        else:
            # 연속이면 점 + lowess 대체: rolling mean (정렬 후)
            d2 = df[[col, "f1_macro"]].dropna().sort_values(col)
            ax.scatter(d2[col], d2["f1_macro"], s=10, alpha=0.5)
            # 간단 롤링 평균
            if len(d2) > 20:
                k = max(5, len(d2)//20)
                roll = d2["f1_macro"].rolling(k, center=True).mean()
                ax.plot(d2[col], roll, linewidth=2)

        ax.set_title(f"f1_macro vs {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("f1_macro")

    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    eff_path = os.path.join(out_dir, "hpo_param_effects.png")
    plt.savefig(eff_path, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved -> {eff_path}")

    # ── 3) Spearman 상관계수(파라미터 ↔ f1_macro) ────────────────
    corr_rows = []
    for col in param_cols:
        try:
            rho = df[[col, "f1_macro"]].corr(method="spearman").iloc[0, 1]
        except Exception:
            rho = np.nan
        corr_rows.append({"param": col, "spearman_w_f1": float(rho)})
    corr_df = pd.DataFrame(corr_rows).sort_values("spearman_w_f1", ascending=False)
    corr_path = os.path.join(out_dir, "hpo_param_spearman.csv")
    corr_df.to_csv(corr_path, index=False, encoding="utf-8-sig")
    print(f"✅ Saved -> {corr_path}")

    # ── 4) 추천 탐색 범위 산출(상위 top_frac 내부 중심 구간) ───────
    # 정수형은 [min, max], 실수형은 [q10, q90] 권장
    rec_rows = []
    for col in param_cols:
        s_top = top[col].dropna()
        if s_top.empty:
            rec_rows.append({"param": col, "recommended_min": None, "recommended_max": None, "dtype": "unknown"})
            continue

        if is_int_series(s_top):
            vmin, vmax = int(s_top.min()), int(s_top.max())
            rec_rows.append({"param": col, "recommended_min": vmin, "recommended_max": vmax, "dtype": "int"})
        else:
            q10, q90 = float(s_top.quantile(0.10)), float(s_top.quantile(0.90))
            rec_rows.append({"param": col, "recommended_min": round(q10, 6), "recommended_max": round(q90, 6), "dtype": "float"})

    rec_df = pd.DataFrame(rec_rows)
    # 보기 좋게 param_ 접두사 제거
    rec_df["param_clean"] = rec_df["param"].str.replace("^param_", "", regex=True)
    rec_df = rec_df[["param", "param_clean", "dtype", "recommended_min", "recommended_max"]]

    rec_csv = os.path.join(out_dir, "hpo_recommended_ranges.csv")
    rec_df.to_csv(rec_csv, index=False, encoding="utf-8-sig")
    print(f"✅ Saved -> {rec_csv}")

    # 요약 txt
    lines = []
    lines.append("===== HPO Recommended Ranges (based on top trials) =====\n")
    lines.append(f"- source: {os.path.basename(hpo_csv)}")
    lines.append(f"- top fraction used: {int(top_frac*100)}% (f1_macro 상위)")
    lines.append("")
    for _, r in rec_df.iterrows():
        lines.append(f"{r['param_clean']:>26s} [{r['dtype']}] : {r['recommended_min']}  ~  {r['recommended_max']}")
    lines.append("")
    lines.append("Tip) 정수형은 그대로 min~max, 실수형은 q10~q90을 다음 탐색 범위로 권장합니다.")
    txt_path = os.path.join(out_dir, "hpo_recommended_ranges.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"✅ Saved -> {txt_path}")

    return {
        "params": param_cols,
        "recommend_csv": rec_csv,
        "corr_csv": corr_path,
        "plots": [dist_path, eff_path]
    }

# ──────────────────────────────
# (선택) 데이터 EDA (기존 기능 유지)
# ──────────────────────────────
def dataset_eda(train_csv: str, out_dir: str):
    ensure_dir(out_dir)
    df = pd.read_csv(train_csv)

    # 0. ID 제거
    for c in ["ID", "id", "Id"]:
        if c in df.columns:
            df = df.drop(columns=[c])

    # 1. Box Plot (수치형, unique > 5, target 제외)
    num_cols = [col for col in df.select_dtypes(include=[np.number]).columns
                if df[col].nunique() > 5 and col != "support_needs"]

    if num_cols:
        plt.figure(figsize=(12, 6), dpi=110)
        flierprops = dict(marker='o', markerfacecolor='red', markersize=5, linestyle='none')
        ax = sns.boxplot(data=df[num_cols], flierprops=flierprops, width=0.4)
        plt.title("box plot for numerical features", fontsize=14)
        plt.xticks(rotation=30, ha="right")

        # 분위수/mean/median 라벨(겹침 완화용 offset)
        for i, col in enumerate(num_cols):
            series = df[col].dropna()
            stats = series.describe(percentiles=[.25, .5, .75])
            mean_val = series.mean()
            q1, med, q3 = stats["25%"], stats["50%"], stats["75%"]
            x_pos = i + 0.35
            ax.text(x_pos, q1, f"Q1={q1:.1f}", color="green", ha="left", va="center", fontsize=7)
            ax.text(x_pos, q3, f"Q3={q3:.1f}", color="purple", ha="left", va="center", fontsize=7)
            ax.text(x_pos, mean_val + 1.0, f"mean={mean_val:.1f}", color="blue", ha="left", va="bottom", fontsize=8)
            ax.text(x_pos, med - 1.0, f"median={med:.1f}", color="red", ha="left", va="top", fontsize=8)

        plt.tight_layout()
        path = os.path.join(out_dir, "boxplot.png")
        plt.savefig(path)
        plt.close()
        print(f"✅ Saved -> {path}")

    # 2. 모든 feature 상위 5개 값 분포 (Bar Plot)
    features = df.columns.tolist()
    n_features = len(features)
    n_cols = 3
    n_rows = int(np.ceil(n_features / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), dpi=120)
    axes = axes.flatten()

    for idx, col in enumerate(features):
        ax = axes[idx]
        vc = df[col].value_counts(normalize=True).head(5) * 100
        vc_count = df[col].value_counts().head(5)
        plot_df = pd.DataFrame({"value": vc.index.astype(str),
                                "count": vc_count.values,
                                "ratio(%)": vc.values.round(2)})

        sns.barplot(data=plot_df, x="value", y="ratio(%)", hue="value",
                    dodge=False, palette="Set2", legend=False, ax=ax)

        ymax = 0.0
        for p, cnt, ratio in zip(ax.patches, plot_df["count"], plot_df["ratio(%)"]):
            x = p.get_x() + p.get_width()/2
            h = p.get_height()
            if h >= 6:
                y, color, va = h * 0.5, "white", "center"
            else:
                y, color, va = h + 0.6, "black", "bottom"
            ax.text(x, y, f"{cnt}\n{ratio:.1f}%", ha="center", va=va, fontsize=9, color=color)
            ymax = max(ymax, h)

        ax.set_title(col, fontsize=11)
        ax.set_xlabel("")
        ax.set_ylabel("Ratio (%)")
        ax.set_ylim(0, max(5, ymax) * 1.12)

    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    path = os.path.join(out_dir, "feature_value_distribution.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved -> {path}")

# ──────────────────────────────
# main
# ──────────────────────────────
def main():
    set_korean_font()

    ap = argparse.ArgumentParser()
    ap.add_argument("--train_input", "-t", default="data/train.csv", help="(선택) train.csv 경로")
    ap.add_argument("--hpo", "-hpo", default=f"results/{MODEL_NAME}_optimization", help="hpo csv 파일 경로 또는 디렉토리")
    ap.add_argument("--output_dir", "-o", default=f"results/eda/{MODEL_NAME}")
    ap.add_argument("--top_frac", type=float, default=0.2, help="상위 비율(다음 탐색 범위 산출에 사용)")
    ap.add_argument("--skip_dataset_eda", action="store_true", help="train.csv EDA를 건너뜀")
    args = ap.parse_args()

    ensure_dir(args.output_dir)

    # 1) HPO 결과 분석 & 다음 탐색 범위 제안
    hpo_csv = find_latest_hpo_csv(args.hpo)
    print(f"[HPO] using: {hpo_csv}")
    analyze_hpo(hpo_csv, os.path.join(args.output_dir, "hpo"), top_frac=args.top_frac)

    # 2) (선택) 기존 데이터 EDA
    if not args.skip_dataset_eda:
        dataset_eda(args.train_input, args.output_dir)

if __name__ == "__main__":
    main()
