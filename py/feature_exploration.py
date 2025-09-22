import os
import random
from datetime import datetime

import numpy as np
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import pandas as pd
from tqdm import tqdm

# ===========================================
# 설정 블록
# ===========================================
SEARCHING_SWITCH = 'optuna'     # 'grid' or 'optuna'
USE_GPU = True
TRIALS = 1000

BASE_FEATURED_DATASET = 'data/cat_train.csv'
target_col = 'support_needs'

MODEL = 'catboost'  # 현재 catboost에 맞춰둠
SEEDS = [45]

USE_CATEGORICAL = True
ADD_FEATURE = "None"

# 👇 catboost_classifier.py와 동일한 metrics 이름 사용
# 'f1_macro', 'class0', 'class1', 'class2', 'bal_acc', 'auprc_macro', 'acc'
METRICS = "f1_macro"

best_metric_overall = 0.0

# categorical feature 제외
df = pd.read_csv(BASE_FEATURED_DATASET)
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
categorical_cols = [c for c in categorical_cols if c not in [target_col, "ID"]]

exclude_cols = categorical_cols + [target_col, "ID"]
num_features = [c for c in df.columns if c not in exclude_cols]


# ===========================================
# 모델 모듈 import
# ===========================================
if MODEL == 'catboost':
    import catboost_classifier as model_module
elif MODEL == 'lightgbm':
    import lightgbm_classifier as model_module
elif MODEL == 'xgboost':
    import xgboost_classifier as model_module
elif MODEL == 'tabnet':
    import tabnet_classifier as model_module
else:
    raise ValueError(f"지원하지 않는 모델: {MODEL}")

# ===========================================
# Feature 생성 로직 (optuna_eda와 동일)
# ===========================================
OPS = ["+", "-", "*", "/", "square", "sqrt", "log"]

def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    denom = b.replace([0, np.inf, -np.inf], np.nan).fillna(1e-6)
    return a / denom

def safe_op(op: str, a: pd.Series, b: pd.Series | None = None) -> pd.Series:
    if op == "+": return (a.fillna(0)) + (b.fillna(0))
    elif op == "-": return (a.fillna(0)) - (b.fillna(0))
    elif op == "*": return (a.fillna(0)) * (b.fillna(0))
    elif op == "/": return _safe_div(a, b)
    elif op == "square": return (a.fillna(0)) ** 2
    elif op == "sqrt": return np.sqrt(np.abs(a.fillna(0)))
    elif op == "log": return np.log1p(np.abs(a.fillna(0)))
    else: return a.fillna(0)

def make_numerical_feature(df: pd.DataFrame, trial):
    numeric_cols = [c for c in df.columns if c not in [target_col, "ID"] and np.issubdtype(df[c].dtype, np.number)]
    if ADD_FEATURE != "None" and ADD_FEATURE in numeric_cols:
        chosen = [ADD_FEATURE]
        others = [c for c in numeric_cols if c != ADD_FEATURE]
        n_extra = trial.suggest_int("n_extra_feats", 1, min(2, len(others))) if others else 0
        if n_extra > 0:
            chosen += random.sample(others, n_extra)
    else:
        n_feats = trial.suggest_int("n_feats", 2, min(3, max(2, len(numeric_cols))))
        chosen = random.sample(numeric_cols, n_feats)
    op = trial.suggest_categorical("num_op", OPS)
    a = df[chosen[0]]
    b = df[chosen[1]] if len(chosen) > 1 else None
    new_series = safe_op(op, a, b)
    formula = f"{op}({','.join(chosen)})"
    return new_series, formula, chosen

def generate_random_formula(full_df, trial):
    # ========================
    # Case 1: ADD_FEATURE 지정
    # ========================
    if ADD_FEATURE != "None":
        # ---- Numerical ADD_FEATURE ----
        if ADD_FEATURE in num_features:
            use_cat = trial.suggest_categorical("use_cat_with_add", [True, False])

            if use_cat and categorical_cols:
                # categorical branch → freq(cat) + ADD_FEATURE
                cat_col = trial.suggest_categorical("categorical_col", categorical_cols)
                cat_series = full_df[cat_col].astype("object").fillna("__NA__")
                freq_map = cat_series.value_counts()
                cat_feature = cat_series.map(freq_map).astype(float).fillna(0.0)

                num_series = full_df[ADD_FEATURE].astype(float)
                op = trial.suggest_categorical("cat_op", ["+", "-", "*", "/"])

                if op == "+":
                    result = cat_feature + num_series
                    expr = f"(freq({cat_col}) + {ADD_FEATURE})"
                elif op == "-":
                    result = cat_feature - num_series
                    expr = f"(freq({cat_col}) - {ADD_FEATURE})"
                elif op == "*":
                    result = cat_feature * num_series
                    expr = f"(freq({cat_col}) * {ADD_FEATURE})"
                elif op == "/":
                    denom = num_series.replace(0, np.nan)
                    result = (cat_feature / denom).fillna(0.0)
                    expr = f"(freq({cat_col}) / {ADD_FEATURE})"

                return result, expr, [cat_col, ADD_FEATURE]

            else:
                # numerical branch → ADD_FEATURE 포함
                n_features = trial.suggest_int("n_features", 1, 4)
                features = random.sample(num_features, n_features)
                if ADD_FEATURE not in features:
                    features[0] = ADD_FEATURE

                result = full_df[features[0]].copy()
                expr = features[0]

                for i in range(1, len(features)):
                    op = trial.suggest_categorical(f"op_{i}", OPS)
                    f = features[i]

                    if op == "square":
                        result = result ** 2
                        expr = f"({expr})^2"
                    elif op == "sqrt":
                        result = np.sqrt(np.abs(result))
                        expr = f"sqrt({expr})"
                    elif op == "log":
                        result = np.log1p(np.abs(result))
                        expr = f"log({expr})"
                    elif op == "+":
                        result = result + full_df[f]
                        expr = f"({expr} + {f})"
                    elif op == "-":
                        result = result - full_df[f]
                        expr = f"({expr} - {f})"
                    elif op == "*":
                        result = result * full_df[f]
                        expr = f"({expr} * {f})"
                    elif op == "/":
                        denom = full_df[f].replace(0, np.nan)
                        result = (result / denom).fillna(0.0)
                        expr = f"({expr} / {f})"

                return result, expr, features

        # ---- Categorical ADD_FEATURE ----
        elif ADD_FEATURE in categorical_cols:
            cat_col = ADD_FEATURE
            cat_series = full_df[cat_col].astype("object").fillna("__NA__")
            freq_map = cat_series.value_counts()
            cat_feature = cat_series.map(freq_map).astype(float).fillna(0.0)

            num_col = trial.suggest_categorical("num_for_cat", num_features)
            op = trial.suggest_categorical("cat_op", ["+", "-", "*", "/"])
            num_series = full_df[num_col].astype(float)

            if op == "+":
                result = cat_feature + num_series
                expr = f"(freq({cat_col}) + {num_col})"
            elif op == "-":
                result = cat_feature - num_series
                expr = f"(freq({cat_col}) - {num_col})"
            elif op == "*":
                result = cat_feature * num_series
                expr = f"(freq({cat_col}) * {num_col})"
            elif op == "/":
                denom = num_series.replace(0, np.nan)
                result = (cat_feature / denom).fillna(0.0)
                expr = f"(freq({cat_col}) / {num_col})"

            return result, expr, [ADD_FEATURE, num_col]

    # ========================
    # Case 2: ADD_FEATURE="None" → 기존 로직
    # ========================
    use_cat = USE_CATEGORICAL and trial.suggest_categorical("use_cat", [True, False])

    if use_cat and categorical_cols:
        cat_col = trial.suggest_categorical("categorical_col", categorical_cols)
        cat_series = full_df[cat_col].astype("object").fillna("__NA__")
        freq_map = cat_series.value_counts()
        cat_feature = cat_series.map(freq_map).astype(float).fillna(0.0)

        num_col = trial.suggest_categorical("num_for_cat", num_features)
        op = trial.suggest_categorical("cat_op", ["+", "-", "*", "/"])
        num_series = full_df[num_col].astype(float)

        if op == "+":
            result = cat_feature + num_series
            expr = f"(freq({cat_col}) + {num_col})"
        elif op == "-":
            result = cat_feature - num_series
            expr = f"(freq({cat_col}) - {num_col})"
        elif op == "*":
            result = cat_feature * num_series
            expr = f"(freq({cat_col}) * {num_col})"
        elif op == "/":
            denom = num_series.replace(0, np.nan)
            result = (cat_feature / denom).fillna(0.0)
            expr = f"(freq({cat_col}) / {num_col})"

        return result, expr, [cat_col, num_col]

    # ---- Numerical 기본 로직 ----
    n_features = trial.suggest_int("n_features", 2, 5)
    features = random.sample(num_features, n_features)

    result = full_df[features[0]].copy()
    expr = features[0]

    for i in range(1, len(features)):
        op = trial.suggest_categorical(f"op_{i}", OPS)
        f = features[i]

        if op == "square":
            result = result ** 2
            expr = f"({expr})^2"
        elif op == "sqrt":
            result = np.sqrt(np.abs(result))
            expr = f"sqrt({expr})"
        elif op == "log":
            result = np.log1p(np.abs(result))
            expr = f"log({expr})"
        elif op == "+":
            result = result + full_df[f]
            expr = f"({expr} + {f})"
        elif op == "-":
            result = result - full_df[f]
            expr = f"({expr} - {f})"
        elif op == "*":
            result = result * full_df[f]
            expr = f"({expr} * {f})"
        elif op == "/":
            denom = full_df[f].replace(0, np.nan)
            result = (result / denom).fillna(0.0)
            expr = f"({expr} / {f})"

    return result, expr, features

# ===========================================
# 결과 저장 준비
# ===========================================
os.makedirs("results/eda/optuna_feature_search", exist_ok=True)
_timestamp = datetime.now().strftime("%d%H%M%S")
_csv_path = os.path.join("results/eda/optuna_feature_search", f"{_timestamp}_trial_results.csv")

with open(_csv_path, "w", encoding="utf-8") as f:
    f.write("trial,formula,used_features,f1_macro,metric\n")

# ===========================================
# Objective 함수
# ===========================================
def objective(trial):
    global best_metric_overall

    df = pd.read_csv(BASE_FEATURED_DATASET)

    if trial.number == 0:
        # === Baseline ===
        new_col_name = None
        formula = "baseline_dataset"
        used_features = []
    else:
        # === 새 feature 생성 ===
        new_series, formula, used_features = generate_random_formula(df, trial)
        new_col_name = f"FEAT__{formula}"
        df[new_col_name] = new_series.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # 임시 저장 (baseline이면 원본만, 아니면 원본+새 feature)
    tmp_path = "tmp__train_with_new_feature.csv"
    df.to_csv(tmp_path, index=False)

    scores, f1_macros = [], []
    for seed in SEEDS:
        result = model_module.train_and_eval(
            train_path=tmp_path,
            seed=seed,
            use_gpu=USE_GPU,
            metrics=METRICS,         # 👈 선택한 metric 전달
            produce_artifacts=False,
            submission=False
        )
        metrics_dict = result["metrics"]
        scores.append(metrics_dict["score"])        # 선택된 metric
        f1_macros.append(metrics_dict["f1_macro"])  # 참고용

    mean_metric = float(np.mean(scores))
    mean_f1_macro = float(np.mean(f1_macros))

    if mean_metric > best_metric_overall:
        best_metric_overall = mean_metric

    # CSV 기록
    with open(_csv_path, "a", encoding="utf-8") as f:
        f.write(f"{trial.number},{formula},{'|'.join(used_features)},"
                f"{mean_f1_macro:.4f},{mean_metric:.4f}\n")

    trial.set_user_attr("formula", formula)
    trial.set_user_attr("metric_score", mean_metric)
    trial.set_user_attr("f1_macro", mean_f1_macro)

    try:
        os.remove(tmp_path)
    except Exception:
        pass

    return mean_metric

# ===========================================
# 실행부
# ===========================================
if __name__ == "__main__":
    if SEARCHING_SWITCH != "optuna":
        raise NotImplementedError("현재 버전은 optuna만 지원합니다.")

    study = optuna.create_study(direction="maximize")

    # tqdm 하나만 열기
    with tqdm(total=TRIALS, desc="Optuna", ncols=180) as pbar:

        def _cb(study, trial):
            best_trial = study.best_trial
            best_num = best_trial.number
            best_f1m = best_trial.user_attrs.get("f1_macro", 0.0)

            formula = trial.user_attrs.get("formula", "")
            score = trial.user_attrs.get("metric_score", 0.0)
            f1m = trial.user_attrs.get("f1_macro", 0.0)

            pbar.set_postfix_str(
                f"best trial is : {best_num}, f1_macro={best_f1m:.4f}, "
                f"Trial {trial.number}: metric={score:.4f}, f1_macro={f1m:.4f}, {formula}"
            )
            pbar.update(1)

        study.optimize(objective, n_trials=TRIALS, callbacks=[_cb])

        pbar.close()

    print("\n===== 탐색 종료 =====")
    print(f"Best Score (metric={METRICS}): {best_metric_overall:.4f}")
    print("Best Params:", study.best_params)
    print(f"Results CSV: {_csv_path}")
