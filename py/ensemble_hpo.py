# meta_model_hpo.py
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, f1_score

# ensemble.py의 OOF 생성 함수와 설정 변수들을 가져옵니다.
from ensemble import get_oof_predictions, base_models, TRAIN_PATH, TEST_PATH, RANDOM_SEED

# ===================================================================
#                      Grid Search 설정 변수
# ===================================================================
# 탐색할 파라미터 그리드를 정의합니다.
PARAM_GRID = [
    # 1. 'l2' 또는 규제 없음을 지원하는 solver 그룹
    {
        'penalty': ['l2', None],
        'solver': ['lbfgs', 'newton-cg', 'sag', 'newton-cholesky'],
        'C': [0.01, 0.1, 1, 10, 100]
    },
    # 2. 'l1', 'l2'를 모두 지원하는 solver 그룹
    {
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga'],
        'C': [0.01, 0.1, 1, 10, 100]
    },
    # 3. 'elasticnet'을 지원하는 solver 그룹
    {
        'penalty': ['elasticnet'],
        'solver': ['saga'],
        'C': [0.01, 0.1, 1, 10, 100],
        'l1_ratio': [0.3, 0.5, 0.7] # elasticnet을 위한 l1 비율
    }
]
# ===================================================================

if __name__ == "__main__":
    # 1. ensemble.py와 동일하게 데이터 로드 및 전처리
    print("데이터를 로드하고 전처리합니다...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    y_target = train_df['support_needs']
    train_df = train_df.drop(columns=['ID', 'support_needs'])
    test_df = test_df.drop(columns=['ID'])

    categorical_cols = train_df.select_dtypes(include=['object']).columns
    train_df = pd.get_dummies(train_df, columns=categorical_cols, drop_first=True)
    test_df = pd.get_dummies(test_df, columns=categorical_cols, drop_first=True)

    train_cols = train_df.columns
    missing_in_test = set(train_cols) - set(test_df.columns)
    for c in missing_in_test: test_df[c] = 0
    extra_in_test = set(test_df.columns) - set(train_cols)
    test_df = test_df.drop(columns=list(extra_in_test))
    test_df = test_df[train_cols]

    # 2. OOF 예측 피처 생성
    # 이 데이터가 메타 모델 튜닝을 위한 학습 데이터가 됩니다.
    oof_train, _ = get_oof_predictions(train_df, y_target, test_df, base_models)

    # 3. Grid Search를 사용한 메타 모델 튜닝
    print("\nGrid Search로 메타 모델 튜닝을 시작합니다...")

    # F1-macro 점수를 기준으로 최적 모델을 찾도록 설정
    f1_scorer = make_scorer(f1_score, average='macro')

    # 메타 모델 초기화
    meta_model = LogisticRegression(random_state=RANDOM_SEED, max_iter=1000)

    # GridSearchCV 설정
    grid_search = GridSearchCV(
        estimator=meta_model,
        param_grid=PARAM_GRID,
        scoring=f1_scorer,
        cv=5,  # OOF 데이터에 대해 5-Fold 교차 검증을 다시 수행하여 안정적인 점수 측정
        n_jobs=-1,
        verbose=1
    )

    # 튜닝 실행
    grid_search.fit(oof_train, y_target)

    # 4. 최종 결과 출력
    print("\n===== Meta Model HPO 완료 (Grid Search) =====")
    print(f"최고 점수 (f1_macro): {grid_search.best_score_:.4f}")
    print("최적 파라미터:")
    for key, value in grid_search.best_params_.items():
        print(f"  - {key}: {value}")