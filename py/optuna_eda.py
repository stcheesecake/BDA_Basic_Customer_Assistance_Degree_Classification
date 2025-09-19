import pandas as pd
import numpy as np
import optuna
import random
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from lightgbm import LGBMClassifier
from tqdm import tqdm
import csv
from datetime import datetime


optuna.logging.set_verbosity(optuna.logging.WARNING)  # Optuna 로그 최소화

# ===============================
# 설정
# ===============================
# 평가 기준 선택: 'F1_MACRO', 'CLASS_0', 'CLASS_1', 'CLASS_2'
USE_CATEGORICAL = True
METRICS = "CLASS_1"
N_TRIALS = 2000

# ===============================
# 1. 데이터 로드
# ===============================
FILE_PATH = "data/total_train.csv"
df = pd.read_csv(FILE_PATH)

target_col = "support_needs"
y = df[target_col]   # ✅ 다중클래스 그대로 사용 (0,1,2)

# categorical feature 제외
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
categorical_cols = [c for c in categorical_cols if c not in [target_col, "ID"]]

exclude_cols = categorical_cols + [target_col, "ID"]
num_features = [c for c in df.columns if c not in exclude_cols]


X = df[num_features].fillna(0)

# ===============================
# 2. 무작위 수식 생성 함수
# ===============================
OPS = ["+", "-", "*", "/", "square", "sqrt", "log"]

def generate_random_formula(full_df, trial):
    if USE_CATEGORICAL:
        use_cat = trial.suggest_categorical("use_cat", [True, False])
    else:
        use_cat = False

    # --- Categorical 분기 ---
    if use_cat and categorical_cols:
        cat_col = trial.suggest_categorical("categorical_col", categorical_cols)

        # NaN 안전한 frequency encoding
        cat_series = full_df[cat_col].astype("object").fillna("__NA__")
        freq_map = cat_series.value_counts()
        cat_feature = cat_series.map(freq_map).astype(float).fillna(0.0)

        # 숫자 피처 하나와 합성
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

        used_features = [cat_col, num_col]
        return result, expr, used_features

    # --- Numerical 분기(기존 로직) ---
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


    # -------------------------------
    # 3. Numerical 합성 선택 (기존 로직)
    # -------------------------------
    n_features = trial.suggest_int("n_features", 2, 5)
    features = random.sample(num_features, n_features)

    result = df[features[0]].copy()
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
            result = result + df[f]
            expr = f"({expr} + {f})"

        elif op == "-":
            result = result - df[f]
            expr = f"({expr} - {f})"

        elif op == "*":
            result = result * df[f]
            expr = f"({expr} * {f})"

        elif op == "/":
            denom = df[f].replace(0, np.nan)
            result = (result / denom).fillna(0)
            expr = f"({expr} / {f})"

    return result, expr, features


# ===============================
# 3. CSV 로깅 설정
# ===============================
SAVE_DIR = "results/eda/optuna_feature_search"
os.makedirs(SAVE_DIR, exist_ok=True)

# 날짜-시간 스탬프 (일-시-분-초)
timestamp = datetime.now().strftime("%d%H%M%S")
CSV_PATH = os.path.join(SAVE_DIR, f"{timestamp}_trial_results.csv")

# 헤더 작성
with open(CSV_PATH, "w", encoding="utf-8-sig") as f:
    f.write("trial,features,formula,f1_class0,f1_class1,f1_class2,f1_macro\n")

# ===============================
# 4. Optuna 목적 함수
# ===============================
pbar = tqdm(total=N_TRIALS, desc="진행률", ncols=100)

def objective(trial):
    new_feature, formula, used_features = generate_random_formula(df, trial)
    X_new = pd.DataFrame({formula: new_feature})  # ✅ 새 feature 단독 사용

    X_train, X_val, y_train, y_val = train_test_split(
        X_new, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=-1,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_val)

    report = classification_report(y_val, preds, output_dict=True, zero_division=0)

    f1_macro = f1_score(y_val, preds, average="macro")
    f1_class0 = report["0"]["f1-score"] if "0" in report else 0
    f1_class1 = report["1"]["f1-score"] if "1" in report else 0
    f1_class2 = report["2"]["f1-score"] if "2" in report else 0

    # 평가 기준 선택
    if METRICS == "F1_MACRO":
        score = f1_macro
    elif METRICS == "CLASS_0":
        score = f1_class0
    elif METRICS == "CLASS_1":
        score = f1_class1
    elif METRICS == "CLASS_2":
        score = f1_class2
    else:
        raise ValueError(f"Unknown METRICS: {METRICS}")

    # CSV 저장
    safe_formula = formula.replace("\n", " ").replace('"', "'")
    safe_features = ";".join(used_features)  # 쉼표 대신 세미콜론

    with open(CSV_PATH, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            trial.number,
            safe_features,
            safe_formula,
            f1_class0,
            f1_class1,
            f1_class2,
            f1_macro
        ])

    # ✅ formula를 trial 속성에 저장 (KeyError 방지)
    trial.set_user_attr("formula", formula)

    # tqdm 업데이트
    past_scores = [t.value for t in study.trials if t.value is not None]
    current_best = max(past_scores + [score]) if past_scores else score
    pbar.set_postfix({"best f1": f"{current_best:.4f}"})
    pbar.update(1)

    return score

# ===============================
# 5. 실행
# ===============================
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=N_TRIALS)
pbar.close()

best_formula = study.best_trial.user_attrs.get("formula", "N/A")
print("Best Score:", study.best_value)
print("Best Formula:", best_formula)
print(f"✅ 모든 trial 결과는 {CSV_PATH} 에 저장되었습니다.")
