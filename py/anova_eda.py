import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway, chi2_contingency
from functools import reduce
import operator
import os




NEW_FEATURE = 'payment_interval, age'
CALC = 'SUM'   # SUM, TIMES, MINUS, DIV 가능
TARGET = 'support_needs'
TRAIN = 'data/test_train.csv'
SAVE = f'results/eda/visualization/{TARGET}'





# --- 폰트 설정 ---
# Windows의 경우 'Malgun Gothic', macOS의 경우 'AppleGothic'으로 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


# ==============================================================================
# 1. 시각화 분석 함수
# ==============================================================================
def plot_numerical_feature(dataframe, feature_name, target_name, save_dir):
    plt.figure(figsize=(12, 7))
    sns.kdeplot(data=dataframe, x=feature_name, hue=target_name,
                fill=True, common_norm=False, palette='viridis')
    plt.title(f'"{feature_name}"의 {TARGET}별 분포', fontsize=15)
    plt.xlabel(feature_name, fontsize=12)
    plt.ylabel('밀도', fontsize=12)
    filename = f"{feature_name}.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path)
    plt.close()

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

def plot_categorical_feature(dataframe, feature_name, target_name, save_dir):
    """
    범주형 피처의 카테고리별 타겟 변수 인원수를 시각화하고 각 막대에 인원수를 표시합니다.
    """
    # 피처와 타겟 변수로 교차표 생성 (실제 인원수)
    ct = pd.crosstab(dataframe[feature_name], dataframe[target_name])

    # 누적 막대그래프 생성
    ax = ct.plot(kind='bar', stacked=True, figsize=(12, 8),
                 colormap='viridis', rot=45)

    # --- 각 막대(patch) 위에 인원수 텍스트 추가 ---
    for p in ax.patches:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy()

        # 인원수가 0보다 클 때만 텍스트 표시
        if height > 0:
            ax.text(x + width / 2,
                    y + height / 2,
                    f'{int(height)}',  # 실제 인원수
                    ha='center',
                    va='center',
                    color='white',
                    fontweight='bold')

    plt.title(f'"{feature_name}"의 카테고리별 {TARGET} 인원수', fontsize=15)
    plt.xlabel(feature_name, fontsize=12)
    plt.ylabel('인원수 (Count)', fontsize=12)
    plt.legend(title=target_name)

    # 파일 이름 변경 (비율 -> 인원수)
    filename = f"{feature_name}.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path)
    plt.close()


# ==============================================================================
# 2. 통계적 분석 함수
# ==============================================================================
def calculate_anova(dataframe, feature_name, target_name):
    groups = [dataframe[dataframe[target_name] == group_val][feature_name]
              for group_val in sorted(dataframe[target_name].unique())]
    f_stat, p_value = f_oneway(*groups)
    return f_stat, p_value


def calculate_chi2(dataframe, feature_name, target_name):
    ct = pd.crosstab(dataframe[feature_name], dataframe[target_name])
    chi2, p, _, _ = chi2_contingency(ct)
    return chi2, p


# ==============================================================================
# 3. 메인 실행부
# ==============================================================================
if __name__ == "__main__":
    # --- 경로 설정 ---
    file_path = TRAIN
    save_dir = SAVE
    target_variable = TARGET

    # --- 저장 폴더 생성 ---
    os.makedirs(save_dir, exist_ok=True)
    print(f"시각화 결과 및 분석 파일은 '{save_dir}' 폴더에 저장됩니다.")

    # --- 데이터 로드 ---
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"오류: '{file_path}' 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
        exit()
    df = create_new_feature(df, NEW_FEATURE, CALC)
    print(f"데이터 로드 완료. Shape: {df.shape}")
    print("-" * 50)

    # --- 피처 타입 분류 ---
    identifier_cols = ['ID']
    features = [col for col in df.columns if col not in [target_variable] + identifier_cols]
    numerical_features = df[features].select_dtypes(include=np.number).columns.tolist()
    categorical_features = df[features].select_dtypes(include=['object', 'category']).columns.tolist()

    # --- 분석 실행 및 결과 저장 ---
    results = []

    print("\n[ 숫자형 피처 분석 중... ]")
    for feature in numerical_features:
        plot_numerical_feature(df, feature, target_variable, save_dir)
        f_stat, p_value = calculate_anova(df, feature, target_variable)
        results.append({'Feature': feature, 'Type': 'Numerical', 'Statistic': f_stat, 'p-value': p_value})

    print("[ 범주형 피처 분석 중... ]")
    for feature in categorical_features:
        plot_categorical_feature(df, feature, target_variable, save_dir)
        chi2, p_value = calculate_chi2(df, feature, target_variable)
        results.append({'Feature': feature, 'Type': 'Categorical', 'Statistic': chi2, 'p-value': p_value})

    print("모든 피처 분석 및 시각화 파일 저장이 완료되었습니다.")

    # --- 최종 결과 및 순위 발표 ---
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by='Statistic', ascending=False).reset_index(drop=True)

    # --- [수정] 결과를 CSV 파일로 저장 ---
    # 1. 피처 중요도 순위 저장
    output_csv_path = os.path.join(save_dir, 'feature_importance_ranking.csv')
    results_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ 피처 순위 결과가 '{output_csv_path}'에 저장되었습니다.")

    # 2. [추가] 범주형 피처별/타겟별 인원수 통계표 저장
    for feature in categorical_features:
        ct = pd.crosstab(df[feature], df[TARGET])
        dist_path = os.path.join(save_dir, f'distribution_{feature}.csv')
        ct.to_csv(dist_path, encoding='utf-8-sig')
        print(f"✅ '{feature}'의 인원수 분포표가 '{dist_path}'에 저장되었습니다.")

    # 3. [추가] 타겟별 숫자 피처 평균값 저장
    mean_values = df.groupby(TARGET)[numerical_features].mean().transpose()
    mean_path = os.path.join(save_dir, 'numerical_features_mean_by_target.csv')
    mean_values.to_csv(mean_path, encoding='utf-8-sig')
    print(f"✅ 타겟별 숫자 피처 평균값이 '{mean_path}'에 저장되었습니다.")

    # ------------------------------------

    # --- 결과를 콘솔에 출력 ---
    print("\n" + "=" * 60)
    print("🏆 타겟 변수 분리 성능 TOP 피처 순위 🏆")
    print("-" * 60)
    print(results_df)
    print("=" * 60)