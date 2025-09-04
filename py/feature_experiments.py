import pandas as pd
import numpy as np
import os

# model.py 스크립트를 불러옵니다.
# 이 코드를 실행하기 전에 model.py가 같은 폴더에 있어야 합니다.
import model

# ===================================================================
#                      사용자 설정 변수
# ===================================================================
# 실험에 사용할 데이터셋 경로를 지정합니다.
# 이전 단계에서 생성한 'train_featured.csv'를 사용합니다.
DATASET_PATH = 'data/train_featured.csv'

# 데이터셋에서 제외할 피처 목록을 지정합니다.
# 예: 원본 피처를 제외하고 새로 만든 피처만으로 성능을 보고 싶을 때
# EXCEPT_FEATURES = ['age', 'tenure', 'frequent']
EXCEPT_FEATURES = ['ID']  # 기본적으로 ID는 제외합니다.

# 실험을 반복할 시드(seed) 목록입니다.
SEEDS = [42, 43, 44, 45, 46]


# ===================================================================


def run_cv_with_seeds(dataset_path, except_features, seeds):
    """
    주어진 데이터셋과 설정으로 여러 시드에 대한 CV 점수를 계산하고 평균을 반환합니다.

    Args:
        dataset_path (str): 학습에 사용할 데이터셋 파일 경로
        except_features (list): 데이터셋에서 제외할 피처 이름 목록
        seeds (list): 실험에 사용할 시드 목록

    Returns:
        float: 모든 시드에 대한 F1-Macro 점수의 평균
    """

    print("===== 피처 실험을 시작합니다 =====")
    print(f"사용 데이터셋: {dataset_path}")
    print(f"제외할 피처: {except_features}")
    print(f"실행 시드: {seeds}")
    print("-" * 30)

    f1_scores = []

    # 원본 데이터프레임 불러오기
    full_df = pd.read_csv(dataset_path)

    # 지정된 피처 제외
    # drop 메서드는 해당 컬럼이 없는 경우 오류를 발생시키므로, 존재하는 컬럼만 제외하도록 처리
    features_to_drop = [col for col in except_features if col in full_df.columns]
    if len(features_to_drop) > 0:
        df = full_df.drop(columns=features_to_drop)
    else:
        df = full_df

    # 실험을 위해 임시 데이터 파일을 생성합니다.
    temp_train_path = 'temp_train_for_experiment.csv'
    df.to_csv(temp_train_path, index=False)

    for seed in seeds:
        print(f"\n>>> 현재 시드: {seed}")

        # model.py의 train_and_eval 함수 호출
        # produce_artifacts=False로 설정하여 결과 파일(모델, 로그 등)이 생성되지 않도록 합니다.
        result = model.train_and_eval(
            train_path=temp_train_path,
            seed=seed,
            produce_artifacts=False,  # 중요: 결과 파일 생성 안 함
            params_dict={'verbose': 0}  # 중요: CatBoost의 학습 로그를 출력하지 않음
        )

        f1_macro = result['metrics']['f1_macro']
        f1_scores.append(f1_macro)
        print(f"시드 {seed}의 F1-Macro 점수: {f1_macro:.4f}")

    # 실험이 끝난 후 임시 파일 삭제
    os.remove(temp_train_path)

    # 최종 평균 점수 계산
    mean_f1_score = np.mean(f1_scores)

    print("-" * 30)
    print(f"\n===== 최종 결과 =====")
    print(f"모든 시드에 대한 평균 F1-Macro CV 점수: {mean_f1_score:.4f}")

    return mean_f1_score


# --- 코드 실행 부분 ---
if __name__ == "__main__":
    run_cv_with_seeds(
        dataset_path=DATASET_PATH,
        except_features=EXCEPT_FEATURES,
        seeds=SEEDS
    )