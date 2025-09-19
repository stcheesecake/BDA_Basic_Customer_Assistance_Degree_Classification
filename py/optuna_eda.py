import pandas as pd
import numpy as np
import optuna
import random
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, balanced_accuracy_score, average_precision_score
from sklearn.preprocessing import label_binarize
from lightgbm import LGBMClassifier
from tqdm import tqdm
import csv
from datetime import datetime

optuna.logging.set_verbosity(optuna.logging.WARNING)  # Optuna 로그 최소화

# ===============================
# 설정
# ===============================
# 평가 기준 선택: 'F1_MACRO', 'CLASS_0', 'CLASS_1', 'CLASS_2', 'F1_HARM', 'BAL_ACC', 'AUPRC_MACRO'
USE_CATEGORICAL = True
METRICS = "F1_WEIGHTED"
N_TRIALS = 2000
ADD_FEATURE = "None"   # "None" → 기존 로직
                       # numerical feature 이름 → 모든 branch에 포함
                       # categorical feature 이름 → categorical branch만 실행

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
pbar = tqdm(total=N_TRIALS, desc="진행률", ncols=150)

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
    elif METRICS == "F1_HARM":
        eps = 1e-9
        score = 3.0 / (1.0/(f1_class0+eps) + 1.0/(f1_class1+eps) + 1.0/(f1_class2+eps))
    elif METRICS == "BAL_ACC":
        score = balanced_accuracy_score(y_val, preds)
    elif METRICS == "AUPRC_MACRO":
        Y = label_binarize(y_val, classes=[0,1,2])
        proba = model.predict_proba(X_val)
        score = average_precision_score(Y, proba, average="macro")
    elif METRICS == "F1_WEIGHTED":
        weights = {0: 1.0, 1: 3.0, 2: 1.5}  # 클래스별 가중치 설정
        score = (
                        f1_class0 * weights[0] +
                        f1_class1 * weights[1] +
                        f1_class2 * weights[2]
                ) / sum(weights.values())
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
    trial.set_user_attr("f1_macro", f1_macro)
    trial.set_user_attr("f1_class1", f1_class1)
    trial.set_user_attr("metric_score", score)

    # tqdm 업데이트
    best_f1_macro = study.best_trial.user_attrs.get("f1_macro", 0.0) if len(study.trials) > 1 else f1_macro
    best_f1_class1 = study.best_trial.user_attrs.get("f1_class1", 0.0) if len(study.trials) > 1 else f1_class1
    best_metric = study.best_trial.user_attrs.get("metric_score", 0.0) if len(study.trials) > 1 else score

    pbar.set_postfix({
        "best f1_macro": f"{best_f1_macro:.4f}",
        "best f1_class1": f"{best_f1_class1:.4f}",
        "metric": f"{best_metric:.4f}"
    })
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
