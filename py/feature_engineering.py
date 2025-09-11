import pandas as pd
import os
import numpy as np
from sklearn.cluster import KMeans

# ===================================================================
#                      사용자 설정 변수
# ===================================================================
# 원본 데이터가 있는 경로를 지정합니다.
ORIGINAL_DATASET_PATH = 'data/test.csv'

# 새로 저장할 파일 이름을 지정합니다.
NEW_DATASET_NAME = '1_test.csv'

# 파일을 저장할 폴더를 지정합니다.
OUTPUT_DIR = 'data/'


# ===================================================================


def add_feature(df):
    """
    원본 데이터프레임에 새로운 피처를 추가하는 함수입니다.
    새로운 피처 아이디어가 생기면 이 함수 내부를 수정하세요.

    Args:
        df (pd.DataFrame): 원본 데이터프레임

    Returns:
        pd.DataFrame: 새로운 피처가 추가된 데이터프레임
    """
    print("새로운 피처 생성을 시작합니다...")

    new_df = df.copy()

    # --- Feature 1: is_older_group ---
    age_median = new_df['age'].median()
    new_df['is_older_group'] = (new_df['age'] > age_median).astype(int)
    print(f"- 'is_older_group' 생성 완료 (기준 나이: {age_median:.1f}세)")

    # --- Feature 2: older_and_member ---
    is_older = (new_df['age'] > age_median)
    is_member = (new_df['subscription_type'] == 'member')
    new_df['older_and_member'] = (is_older & is_member).astype(int)
    print("- 'older_and_member' 생성 완료")

    # --- Feature 3: is_low_frequency ---
    freq_q1 = new_df['frequent'].quantile(0.25)
    new_df['is_low_frequency'] = (new_df['frequent'] <= freq_q1).astype(int)
    print(f"- 'is_low_frequency' 생성 완료 (기준 빈도: {freq_q1:.1f})")

    # --- Feature 4: vip_inactive ---
    is_premium = new_df['subscription_type'].isin(['vip', 'plus'])
    is_inactive = (new_df['is_low_frequency'] == 1)
    new_df['vip_inactive'] = (is_premium & is_inactive).astype(int)
    print("- 'vip_inactive' 생성 완료")

    # --- Feature 5: new_inactive ---
    tenure_q1 = new_df['tenure'].quantile(0.25)
    is_new = (new_df['tenure'] <= tenure_q1)
    new_df['new_inactive'] = (is_new & is_inactive).astype(int)
    print(f"- 'new_inactive' 생성 완료 (기준 가입 기간: {tenure_q1:.1f})")

    # ==============================================================
    # 추가 Feature 9개
    # ==============================================================

    # 1. is_long_contract (>=90)
    new_df['is_long_contract'] = (new_df['contract_length'] >= 90).astype(int)
    print("- 'is_long_contract' 생성 완료 (기준 계약기간: 90일)")

    # 2. is_high_payment_interval (>= Q3)
    pay_q3 = new_df['payment_interval'].quantile(0.75)
    new_df['is_high_payment_interval'] = (new_df['payment_interval'] >= pay_q3).astype(int)
    print(f"- 'is_high_payment_interval' 생성 완료 (기준 결제주기: {pay_q3:.1f})")

    # 3. is_high_interaction (>= Q3)
    inter_q3 = new_df['after_interaction'].quantile(0.75)
    new_df['is_high_interaction'] = (new_df['after_interaction'] >= inter_q3).astype(int)
    print(f"- 'is_high_interaction' 생성 완료 (기준 상호작용: {inter_q3:.1f})")

    # 4. freq_per_tenure
    new_df['freq_per_tenure'] = (new_df['frequent'] / new_df['tenure']).replace([float('inf'), -float('inf')],
                                                                                0).fillna(0)
    print("- 'freq_per_tenure' 생성 완료")

    # 5. interaction_per_freq
    new_df['interaction_per_freq'] = (new_df['after_interaction'] / new_df['frequent']).replace(
        [float('inf'), -float('inf')], 0).fillna(0)
    print("- 'interaction_per_freq' 생성 완료")

    # 6. payment_per_freq
    new_df['payment_per_freq'] = (new_df['payment_interval'] / new_df['frequent']).replace(
        [float('inf'), -float('inf')], 0).fillna(0)
    print("- 'payment_per_freq' 생성 완료")

    # 7. short_tenure_high_interval
    tenure_q1 = new_df['tenure'].quantile(0.25)
    new_df['short_tenure_high_interval'] = (
                (new_df['tenure'] <= tenure_q1) & (new_df['payment_interval'] >= pay_q3)).astype(int)
    print("- 'short_tenure_high_interval' 생성 완료")

    # 8. older_low_contract
    age_median = new_df['age'].median()
    new_df['older_low_contract'] = ((new_df['age'] >= age_median) & (new_df['contract_length'] == 30)).astype(int)
    print("- 'older_low_contract' 생성 완료")

    # 9. vip_low_interaction
    inter_q1 = new_df['after_interaction'].quantile(0.25)
    is_premium = new_df['subscription_type'].isin(['vip', 'plus'])
    new_df['vip_low_interaction'] = (is_premium & (new_df['after_interaction'] <= inter_q1)).astype(int)
    print(f"- 'vip_low_interaction' 생성 완료 (기준 상호작용: {inter_q1:.1f})")


    # ==============================================================
    # 추가 Feature 10개
    # ==============================================================

    # 3. interaction_rate
    new_df['interaction_rate'] = new_df['after_interaction'] / (
        new_df['after_interaction'] + new_df['frequent']
    ).replace(0, np.nan)
    new_df['interaction_rate'] = new_df['interaction_rate'].fillna(0)
    print("- 'interaction_rate' 생성 완료")

    # 4. contract_ratio (계약기간 ÷ 결제주기)
    new_df['contract_ratio'] = (
        new_df['contract_length'] / new_df['payment_interval']
    ).replace([np.inf, -np.inf], 0).fillna(0)
    print("- 'contract_ratio' 생성 완료")

    # 5. payment_freq_alignment (결제주기가 7의 배수인지)
    new_df['payment_freq_alignment'] = (new_df['payment_interval'] % 7 == 0).astype(int)
    print("- 'payment_freq_alignment' 생성 완료")

    # 6. renewal_pressure (계약길이 대비 사용기간)
    new_df['renewal_pressure'] = (
        new_df['contract_length'] / new_df['tenure']
    ).replace([np.inf, -np.inf], 0).fillna(0)
    print("- 'renewal_pressure' 생성 완료")

    # 7. subscription_code (member=0, plus=1, vip=2)
    sub_map = {'member': 0, 'plus': 1, 'vip': 2}
    new_df['subscription_code'] = new_df['subscription_type'].map(sub_map).fillna(-1).astype(int)
    print("- 'subscription_code' 생성 완료")

    # 8. gender_age_group (F-young, F-old, M-young, M-old → 0~3)
    age_median = new_df['age'].median()
    def encode_gender_age(row):
        if row['gender'] == 'F' and row['age'] < age_median:
            return 0
        elif row['gender'] == 'F' and row['age'] >= age_median:
            return 1
        elif row['gender'] == 'M' and row['age'] < age_median:
            return 2
        else:
            return 3
    new_df['gender_age_group'] = new_df.apply(encode_gender_age, axis=1)
    print("- 'gender_age_group' 생성 완료")

    # 9. usage_cluster (frequent, after_interaction, payment_interval 기준 KMeans=3)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    usage_features = new_df[['frequent', 'after_interaction', 'payment_interval']].fillna(0)
    new_df['usage_cluster'] = kmeans.fit_predict(usage_features)
    print("- 'usage_cluster' 생성 완료")






    print("모든 피처 생성이 완료되었습니다.")

    return new_df


# --- 코드 실행 부분 ---

# 저장할 디렉토리 생성
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

output_path = os.path.join(OUTPUT_DIR, NEW_DATASET_NAME)

# 원본 데이터셋 불러오기
print(f"원본 데이터셋을 불러옵니다: {ORIGINAL_DATASET_PATH}")
original_df = pd.read_csv(ORIGINAL_DATASET_PATH)

# 함수를 호출하여 새로운 피처가 추가된 데이터셋 생성
featured_df = add_feature(original_df)

# 새로운 데이터셋 저장
print(f"새로운 피처가 추가된 데이터셋을 저장합니다: {output_path}")
featured_df.to_csv(output_path, index=False)

print("\n작업 완료!")
print("생성된 데이터셋의 상위 5개 행:")
print(featured_df.head())