import pandas as pd
import os
import numpy as np
from sklearn.cluster import KMeans

# ===================================================================
#                      사용자 설정 변수
# ===================================================================
# 원본 데이터가 있는 경로를 지정합니다.
ORIGINAL_DATASET_PATH = 'data/original_train.csv'

# 새로 저장할 파일 이름을 지정합니다.
NEW_TRAIN_NAME = 'cat_train.csv'
NEW_TEST_NAME = 'cat_test.csv'

# 파일을 저장할 폴더를 지정합니다.
OUTPUT_DIR = 'data/'

add_feature_name_list = 'is_older_group, new_inactive, is_high_interaction, freq_per_tenure, interaction_per_freq, payment_per_freq, older_low_contract, vip_low_interaction, interaction_rate, contract_ratio, renewal_pressure, gender_age_group, usage_cluster'


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

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print("=" * 50)
print(f"Train 데이터를 처리합니다: {ORIGINAL_DATASET_PATH}")
df_train = pd.read_csv(ORIGINAL_DATASET_PATH)
original_train_cols = df_train.columns.tolist()
df_train_featured = add_feature(df_train)

test_path = ORIGINAL_DATASET_PATH.replace("train.csv", "test.csv")
if os.path.exists(test_path):
    print("\n" + "=" * 50)
    print(f"Test 데이터를 처리합니다: {test_path}")
    df_test = pd.read_csv(test_path)
    original_test_cols = df_test.columns.tolist()
    df_test_featured = add_feature(df_test)
else:
    df_test_featured = None
    print(f"\n경고: Test 파일({test_path})을 찾을 수 없습니다.")

# --- [요청사항] 피처 선택 로직 추가 ---
# add_feature_name_list가 비어있지 않은 문자열일 경우에만 실행
if add_feature_name_list and isinstance(add_feature_name_list, str):
    print("\n" + "=" * 50)
    print("지정된 피처만 선택하여 최종 파일을 생성합니다...")

    # 쉼표로 구분된 문자열을 공백 제거 후 리스트로 변환
    selected_features = [feat.strip() for feat in add_feature_name_list.split(',')]

    # Train 데이터: 원본 컬럼 + 선택된 신규 피처
    final_train_cols = original_train_cols + [feat for feat in selected_features if feat in df_train_featured.columns]
    df_train_featured = df_train_featured[list(dict.fromkeys(final_train_cols))]  # 중복 제거하며 순서 유지
    print(f"- Train 데이터 컬럼 수: {len(df_train_featured.columns)}")

    # Test 데이터: 원본 컬럼 + 선택된 신규 피처
    if df_test_featured is not None:
        final_test_cols = original_test_cols + [feat for feat in selected_features if feat in df_test_featured.columns]
        df_test_featured = df_test_featured[list(dict.fromkeys(final_test_cols))]
        print(f"- Test 데이터 컬럼 수: {len(df_test_featured.columns)}")

# --- 파일 저장 (원본 로직) ---
train_save_path = os.path.join(OUTPUT_DIR, NEW_TRAIN_NAME)
df_train_featured.to_csv(train_save_path, index=False, encoding='utf-8-sig')
print(f"\n✅ 새로운 Train 파일이 저장되었습니다: {train_save_path}")

if df_test_featured is not None:
    test_save_path = os.path.join(OUTPUT_DIR, NEW_TEST_NAME)
    df_test_featured.to_csv(test_save_path, index=False, encoding='utf-8-sig')
    print(f"✅ 새로운 Test 파일이 저장되었습니다: {test_save_path}")

print("\n모든 작업이 완료되었습니다.")