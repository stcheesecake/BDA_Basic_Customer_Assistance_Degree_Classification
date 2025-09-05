import pandas as pd
import os

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
    # age_median = new_df['age'].median()
    # new_df['is_older_group'] = (new_df['age'] > age_median).astype(int)
    # print(f"- 'is_older_group' 생성 완료 (기준 나이: {age_median:.1f}세)")

    # # --- Feature 2: older_and_member ---
    # is_older = (new_df['age'] > age_median)
    # is_member = (new_df['subscription_type'] == 'member')
    # new_df['older_and_member'] = (is_older & is_member).astype(int)
    # print("- 'older_and_member' 생성 완료")

    # --- Feature 3: is_low_frequency ---
    freq_q1 = new_df['frequent'].quantile(0.25)
    new_df['is_low_frequency'] = (new_df['frequent'] <= freq_q1).astype(int)
    print(f"- 'is_low_frequency' 생성 완료 (기준 빈도: {freq_q1:.1f})")

    # --- Feature 4: vip_inactive ---
    is_premium = new_df['subscription_type'].isin(['vip', 'plus'])
    is_inactive = (new_df['is_low_frequency'] == 1)
    new_df['vip_inactive'] = (is_premium & is_inactive).astype(int)
    print("- 'vip_inactive' 생성 완료")

    # # --- Feature 5: new_inactive ---
    # tenure_q1 = new_df['tenure'].quantile(0.25)
    # is_new = (new_df['tenure'] <= tenure_q1)
    # new_df['new_inactive'] = (is_new & is_inactive).astype(int)
    # print(f"- 'new_inactive' 생성 완료 (기준 가입 기간: {tenure_q1:.1f})")

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