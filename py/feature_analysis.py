import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# matplotlib에서 한글 폰트가 깨지지 않도록 설정합니다.
# 로컬 환경에 맞는 폰트 이름을 지정해야 합니다. (예: 'Malgun Gothic' for Windows)
try:
    plt.rcParams['font.family'] = 'NanumGothic'
except:
    plt.rcParams['font.family'] = 'Malgun Gothic' # Windows 기본 폰트
plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지

# 분석할 데이터를 불러옵니다.
# 사용자의 환경에 맞게 파일 경로를 지정해주세요. (예: 'data/train.csv')
df = pd.read_csv('data/train.csv')

# 결과를 저장할 폴더를 생성합니다.
output_dir = 'results/eda/feature_engineering/'
os.makedirs(output_dir, exist_ok=True)

# --- 1. 상관관계 분석 ---
numerical_df = df.select_dtypes(include=['float64', 'int64'])
corr_matrix = numerical_df.corr()

plt.figure(figsize=(12, 9))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('숫자형 피처 간의 상관관계 히트맵', fontsize=16)

# [수정-1] tight_layout()을 savefig 전에 호출하여 잘림 현상을 방지합니다.
plt.tight_layout()

plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
plt.show()


# --- 2. 시각화 분석 (값 표시 기능 추가) ---
fig, axes = plt.subplots(1, 3, figsize=(22, 7))
features_to_plot = ['tenure', 'frequent', 'age']
titles = ['가입 기간(tenure)', '월 평균 이용 빈도(frequent)', '나이(age)']

# [수정-2] 각 subplot에 대해 반복하면서 박스 플롯을 그리고 값을 표시합니다.
for ax, feature, title in zip(axes, features_to_plot, titles):
    sns.boxplot(x='support_needs', y=feature, data=df, ax=ax)
    ax.set_title(f'지원 필요도에 따른 {title} 분포', fontsize=14)
    ax.set_xlabel('지원 필요도 (0:낮음, 1:중간, 2:높음)')
    ax.set_ylabel(title)

    # 각 그룹(0, 1, 2)별로 통계값 계산
    for i in range(3):
        group_data = df[df['support_needs'] == i][feature]
        q1 = group_data.quantile(0.25)
        median = group_data.median()
        q3 = group_data.quantile(0.75)

        # 박스 위에 Q3, Median, Q1 값을 텍스트로 추가
        # x좌표는 그룹의 위치(0, 1, 2), y좌표는 각 통계값
        ax.text(i, q3, f'Q3: {q3:.1f}', ha='center', va='bottom', fontsize=9, color='blue')
        ax.text(i, median, f'Median: {median:.1f}', ha='center', va='bottom', fontsize=9, color='red')
        ax.text(i, q1, f'Q1: {q1:.1f}', ha='center', va='top', fontsize=9, color='blue')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'visual_analysis_boxplots_with_values.png'))
plt.show()