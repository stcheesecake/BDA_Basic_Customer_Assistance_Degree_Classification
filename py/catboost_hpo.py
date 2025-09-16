# catboost_hpo.py
# -*- coding: utf-8 -*-

import os
import csv
import json
import argparse
from datetime import datetime
import contextlib
import io
import catboost_classifier  # 수정된 catboost_classifier.py를 임포트
import numpy as np
import pandas as pd
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from optuna.samplers import TPESampler
from tqdm import tqdm

N_TRIALS = 6000

# ───────────────────────── 검색 범위 (원본 구조 유지) ─────────────────────────
SEARCH_SPACE = dict(
    iterations=("int", 1400, 1400, 100),
    learning_rate=("float", 0.01, 0.03, 0.001),
    depth=("int", 8, 10, 1),
    l2_leaf_reg=("float", 18.0, 20.0, 1.0),
    border_count=("int", 200, 300, 1),
    random_strength=("float", 0.1, 0.8, 0.01),
    bagging_temperature=("float", 0.3, 0.5, 0.01),
    # [요청사항] 클래스별 가중치 탐색 범위 추가 (start, end, step)
    weights_0=("float", 2.7, 4.0, 0.01),
    weights_1=("float", 4.5, 8.0, 0.01),
    weights_2=("float", 3.5, 5.0, 0.01),
)


# ───────────────────────── 인자 정의 (원본 구조 유지) ─────────────────────────
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", default="data/cat_train.csv")
    ap.add_argument("--save_dir", default="results/catboost_optimization")
    ap.add_argument("--trials", type=int, default=N_TRIALS)
    ap.add_argument("--use_gpu", type=bool, default=True)
    ap.add_argument("--valid_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=45)
    return ap.parse_args()


# ───────────────────────── Objective 함수 (수정) ─────────────────────────
def objective(trial, args, csv_path):
    # 하이퍼파라미터 샘플링 (원본 방식 유지)
    params = {}
    for name, (param_type, low, high, step) in SEARCH_SPACE.items():
        if param_type == "int":
            params[name] = trial.suggest_int(name, low, high, step=step)
        else:  # float
            params[name] = trial.suggest_float(name, low, high, step=step)

    # [요청사항] 샘플링된 가중치 값을 별도로 추출
    w0 = params.pop('weights_0')
    w1 = params.pop('weights_1')
    w2 = params.pop('weights_2')

    # catboost_classifier.py에 전달할 weights 문자열 생성 (json.dumps 사용)
    weights_str = json.dumps([w0, w1, w2])

    # 최종적으로 classifier에 전달할 파라미터 딕셔너리 생성
    classifier_params = params.copy()  # weights가 제외된 파라미터 복사
    classifier_params['weights'] = weights_str  # 문자열로 변환된 가중치 추가
    classifier_params['train_path'] = args.train_path
    classifier_params['use_gpu'] = args.use_gpu
    classifier_params['seed'] = args.seed

    # catboost_classifier.train_and_eval 호출 (무음 처리)
    # **kwargs를 사용하여 파라미터를 동적으로 전달
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        result = catboost_classifier.train_and_eval(**classifier_params)

    f1_macro = result['metrics']['f1_macro']

    # CSV에 즉시 기록
    # [요청사항] trial.params에서 직접 weight 값을 가져와 기록
    log_params = trial.params
    row = [trial.number] + [log_params.get(k) for k in SEARCH_SPACE.keys()] + [f1_macro]
    with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(row)

    return f1_macro


# ───────────────────────── 메인 함수 (원본 구조 유지) ─────────────────────────
def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    now = datetime.now().strftime('%y%m%d_%H%M%S')
    csv_path = os.path.join(args.save_dir, f"{now}_catboost_hpo.csv")

    header = ["trial"] + list(SEARCH_SPACE.keys()) + ["f1_macro"]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(header)

    sampler = TPESampler(seed=args.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    # [✅ 수정] 최고 점수와 trial 번호를 추적할 변수 초기화
    best_f1 = -1.0
    best_trial_num = -1

    # [✅ 수정] tqdm 진행률 표시줄 로직 변경
    with tqdm(total=args.trials, desc="CatBoost HPO") as pbar:
        for i in range(args.trials):
            trial = study.ask()
            f1_macro = objective(trial, args, csv_path)
            study.tell(trial, f1_macro)

            # 최고 점수 갱신
            if f1_macro > best_f1:
                best_f1 = f1_macro
                best_trial_num = trial.number

            # 진행률 표시줄의 설명(description) 업데이트
            pbar.set_description(
                f"CatBoost HPO | best trial = {best_trial_num}, f1-macro = {best_f1:.4f}"
            )
            pbar.update(1)

    # 최적화 결과 출력
    print(f"\n✅ HPO가 완료되었습니다. ({csv_path})")
    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    best_trial = study.best_trial
    print(f"  Value: {best_trial.value:.4f}")
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")


if __name__ == "__main__":
    main()