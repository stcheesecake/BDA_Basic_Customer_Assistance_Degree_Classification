import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, chi2_contingency
import os
from functools import reduce
import operator

# --- 사용자 설정 변수 ---
NEW_FEATURE = 'is_high_payment_interval, is_high_interaction'
CALC = 'TIMES'
TARGET_CLASS = 1
SAVE_DIR = f'results/eda/visualization/class_{TARGET_CLASS}_separation'
FILE_PATH = 'data/total_train.csv'

# --- 폰트 설정 ---
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


# ==============================================================================
# 1. 분석용 함수들
# ==============================================================================
def create_new_feature(df, feature_str, calc_method):
    if not feature_str or not calc_method:
        return df

    features_to_calc = [f.strip() for f in feature_str.split(',')]

    for feature in features_to_calc:
        if feature not in df.columns:
            print(f"경고: '{feature}' 피처가 없어 신규 피처 생성을 건너뜁니다.")
            return df
        df[feature] = pd.to_numeric(df[feature], errors='coerce').fillna(0)

    new_feature_name = f"{calc_method.upper()}({','.join(features_to_calc)})"
    print(f"신규 피처 '{new_feature_name}'를 생성합니다...")

    if calc_method.upper() == 'SUM':
        df[new_feature_name] = df[features_to_calc].sum(axis=1)
    elif calc_method.upper() == 'TIMES':
        df[new_feature_name] = df[features_to_calc].product(axis=1)
    elif calc_method.upper() == 'MINUS':
        df[new_feature_name] = reduce(operator.sub, [df[f] for f in features_to_calc])
    elif calc_method.upper() == 'DIV':
        temp_df = df[features_to_calc].replace(0, np.nan)
        df[new_feature_name] = reduce(operator.truediv, [temp_df[f] for f in features_to_calc])
        df[new_feature_name].fillna(0, inplace=True)

    return df


def plot_and_analyze(df, save_dir, target_class):
    binary_target_name = f'is_class_{target_class}'
    df[binary_target_name] = df['support_needs'].apply(lambda x: 1 if x == target_class else 0)

    identifier_cols = ['ID', 'support_needs']
    features = [col for col in df.columns if col not in identifier_cols + [binary_target_name]]
    numerical_features = df[features].select_dtypes(include=np.number).columns.tolist()
    categorical_features = df[features].select_dtypes(include=['object', 'category']).columns.tolist()

    print("\n" + "-" * 50)
    print("[진단] 최종적으로 다음 피처들을 분석합니다:")
    print(f"  - 숫자형: {numerical_features}")
    print("-" * 50 + "\n")

    results = []

    # 숫자형 피처 분석
    for feature in numerical_features:
        print(f"  [분석 시도] '{feature}' ...")
        plt.figure(figsize=(12, 7))
        sns.kdeplot(data=df, x=feature, hue=binary_target_name, fill=True, common_norm=False, palette='cividis')
        plt.title(f'"{feature}"의 Class {target_class} vs Rest 분포', fontsize=15)
        safe_feature_name = feature.replace('(', '_').replace(')', '').replace(',', '')
        save_path = os.path.join(save_dir, f"{safe_feature_name}.png")
        plt.savefig(save_path)
        plt.close()

        stat, p_value = ttest_ind(df[df[binary_target_name] == 1][feature], df[df[binary_target_name] == 0][feature],
                                  equal_var=False, nan_policy='omit')
        results.append({'Feature': feature, 'Type': 'Numerical', 'Statistic (t-stat)': stat, 'p-value': p_value})

    # 범주형 피처 분석 (이 코드에서는 범주형이 없지만, 혹시 모를 경우를 위해 유지)
    for feature in categorical_features:
        print(f"  [분석 시도] '{feature}' ...")
        # 시각화 및 저장 로직 추가...
        stat, p_value = chi2_contingency(pd.crosstab(df[feature], df[binary_target_name]))[0:2]
        results.append({'Feature': feature, 'Type': 'Categorical', 'Statistic (Chi2)': stat, 'p-value': p_value})

    return pd.DataFrame(results)


# ==============================================================================
# 3. 메인 실행부
# ==============================================================================
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"분석 결과는 '{SAVE_DIR}' 폴더에 저장됩니다.")

    # 1. 원본 데이터 로드
    df_original = pd.read_csv(FILE_PATH)

    # 2. 신규 피처 생성 (df가 이 단계에서 변경됨)
    df_modified = create_new_feature(df_original, NEW_FEATURE, CALC)

    # 3. 변경된 df를 사용해 분석 함수 호출
    results_df = plot_and_analyze(df_modified, SAVE_DIR, TARGET_CLASS)

    print("\n분석 완료.")

    results_df['abs_statistic'] = results_df.iloc[:, 2].abs()
    results_df = results_df.sort_values(by='abs_statistic', ascending=False).drop(columns='abs_statistic')

    output_csv_path = os.path.join(SAVE_DIR, f'feature_ranking_for_class_{TARGET_CLASS}.csv')
    results_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"✅ Class {TARGET_CLASS} 분리 성능 순위가 '{output_csv_path}'에 저장되었습니다.")

    print("\n" + "=" * 60)
    print(f"🏆 Class {TARGET_CLASS} 분리 성능 TOP 피처 순위 🏆")
    print("-" * 60)
    print(results_df)
    print("=" * 60)


if __name__ == "__main__":
    main()