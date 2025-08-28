# spearman_analysis_plus.py
# -*- coding: utf-8 -*-
"""
bo_trials.csv를 읽어 선택한 METRIC(예: balacc/f1/auc)에 대해
- 파라미터별 Spearman rho 표 (상세: overall min/median/max, 상위25% P10/P90, 제안범위)
- HEATMAP(파라미터 × 모든 지표의 Spearman)
- |rho|가 가장 큰 파라미터 산점도
- 상위 3개 파라미터 분포 박스플롯(전체 vs 상위25%)
- 제안 범위를 PARAMS_CONFIG 형태 JSON으로 저장
을 생성합니다.
"""

import os, json, math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===== 설정값(최상단) =====
METRIC   = "balacc"             # 'balacc' | 'f1' | 'auc' 등 CSV에 있는 지표
CSV_PATH = "results/optimization/bo_trials.csv"      # HPO에서 누적 저장된 CSV
OUT_DIR  = "results/eda"   # 산출물 저장 폴더
TOP_Q    = 0.75                 # 상위 25% 기준선
# =========================

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

# 파라미터/지표 컬럼 식별
param_cols = [c for c in df.columns if c.startswith("param_")]
metric_candidates = ["balacc","f1","auc","precision","recall","acc0","acc1","youden","score"]
metrics = [m for m in metric_candidates if m in df.columns]
if METRIC not in metrics:
    raise ValueError(f"METRIC '{METRIC}' not found. Available: {metrics}")

# 숫자 변환
for c in param_cols + metrics:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# 상위 25% 필터
cut = df[METRIC].quantile(TOP_Q)
top = df[df[METRIC] >= cut].copy()

# 상세표: Spearman rho + 전체/상위25% 분포 + 추천 범위
rows = []
for c in param_cols:
    x = df[c]
    rho = x.corr(df[METRIC], method="spearman")
    rows.append({
        "param": c.replace("param_",""),
        "spearman_rho": float(rho) if rho==rho else np.nan,
        "overall_min": float(np.nanmin(x)),
        "overall_median": float(np.nanmedian(x)),
        "overall_max": float(np.nanmax(x)),
        "top25_p10": float(np.nanpercentile(top[c], 10)) if not top[c].dropna().empty else np.nan,
        "top25_p90": float(np.nanpercentile(top[c], 90)) if not top[c].dropna().empty else np.nan,
    })
table = pd.DataFrame(rows).sort_values("spearman_rho", ascending=False, ignore_index=True)

def nice_step(lo, hi, is_int):
    rng = hi - lo
    if rng <= 0 or np.isnan(rng):
        return 1 if is_int else 0.01
    if is_int:
        return max(1, int(round(rng/10)))
    raw = rng/12
    if raw >= 1:   st = round(raw, 1)
    elif raw >= .1: st = round(raw, 2)
    elif raw >= .01: st = round(raw, 3)
    else:          st = round(raw, 4)
    return max(st, 1e-4)

suggestions = {}
for _, r in table.iterrows():
    name = r["param"]; col = "param_" + name
    s_all = df[col]
    is_int = pd.api.types.is_integer_dtype(s_all.dropna().astype(float)) and (s_all.dropna() % 1 == 0).all()
    lo = np.nanmax([r["overall_min"], r["top25_p10"]])
    hi = np.nanmin([r["overall_max"], r["top25_p90"]])
    if np.isnan(lo) or np.isnan(hi) or hi <= lo:
        lo, hi = float(r["overall_min"]), float(r["overall_max"])
    st = nice_step(lo, hi, is_int)
    if is_int:
        lo, hi, st = int(round(lo)), int(round(hi)), int(max(1, round(st)))
    suggestions[name] = [lo, hi, st]

table["suggest_lo"]   = table["param"].map(lambda n: suggestions[n][0])
table["suggest_hi"]   = table["param"].map(lambda n: suggestions[n][1])
table["suggest_step"] = table["param"].map(lambda n: suggestions[n][2])

# 저장: 상세표 + 추천 JSON
out_csv = os.path.join(OUT_DIR, f"spearman_{METRIC}_detailed.csv")
table.to_csv(out_csv, index=False, encoding="utf-8-sig")
with open(os.path.join(OUT_DIR, f"narrowed_params_from_{METRIC}.json"), "w", encoding="utf-8") as f:
    json.dump(suggestions, f, ensure_ascii=False, indent=2)

# HEATMAP (params × metrics Spearman)
H = np.zeros((len(param_cols), len(metrics)), dtype=float)
for i, pc in enumerate(param_cols):
    for j, m in enumerate(metrics):
        H[i, j] = df[pc].corr(df[m], method="spearman")

plt.figure()
plt.imshow(H, aspect="auto")  # 기본 colormap (지정 X)
plt.xticks(ticks=np.arange(len(metrics)), labels=metrics, rotation=45, ha="right")
plt.yticks(ticks=np.arange(len(param_cols)), labels=[c.replace("param_","") for c in param_cols])
plt.title("Spearman heatmap (params × metrics)")
plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "spearman_heatmap.png"))
plt.close()

# 상위 |rho| 파라미터 산점도
top_param = table.iloc[0]["param"]
x = df["param_"+top_param]; y = df[METRIC]
plt.figure()
plt.scatter(x, y, alpha=0.6)
plt.xlabel(top_param); plt.ylabel(METRIC)
plt.title(f"{top_param} vs {METRIC} (Spearman={table.iloc[0]['spearman_rho']:.3f})")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f"scatter_{top_param}_vs_{METRIC}.png"))
plt.close()

# 상위 3개 파라미터: 전체 vs 상위25% 박스플롯 (각각 개별 파일)
top3 = table.head(3)["param"].tolist()
for p in top3:
    col = "param_"+p
    data = [df[col].dropna().values, top[col].dropna().values]
    labels = ["overall", "top25"]
    plt.figure()
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.title(f"Distribution of {p}: overall vs top25({METRIC})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"box_{p}.png"))
    plt.close()

print(f"[Saved] table -> {out_csv}")
print(f"[Saved] heatmap -> {os.path.join(OUT_DIR, 'spearman_heatmap.png')}")
print(f"[Saved] config -> {os.path.join(OUT_DIR, f'narrowed_params_from_{METRIC}.json')}")