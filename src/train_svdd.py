"""SVDD(Support Vector Data Description) 학습/평가 CLI.

기본 사용법:
    python -m src.train_svdd \
        --train-csv ./data/all_train.csv \
        --valid-csv ./data/valid.csv \
        --test-csv ./data/all_test.csv

--search 플래그를 주면 검증셋 F1 기준으로 (C, gamma) grid search를 수행한다.
지정하지 않으면 원 논문/노트북 설정(C=1.0, gamma='auto')으로 바로 학습한다.
"""
from __future__ import annotations

import argparse

import numpy as np
from sklearn.metrics import f1_score

from src.data.dataset import load_and_scale
from src.models.svdd import SVDD, labels_to_pm1, predictions_to_binary
from src.utils import compute_metrics, get_threshold, roc_auc, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="SVDD 학습/평가")
    p.add_argument("--train-csv", type=str, default="./data/all_train.csv")
    p.add_argument("--valid-csv", type=str, default="./data/valid.csv")
    p.add_argument("--test-csv", type=str, default="./data/all_test.csv")
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--gamma", type=str, default="auto", help="'auto', 'scale' 또는 float 문자열")
    p.add_argument("--search", action="store_true", help="검증셋 F1 기준 (C, gamma) grid search 수행")
    p.add_argument("--C-list", type=float, nargs="+", default=[0.5, 0.9, 1.0])
    p.add_argument("--gamma-list", type=float, nargs="+", default=[1e-4, 1e-5, 1e-6, 1e-7])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _parse_gamma(raw: str):
    if raw in ("auto", "scale"):
        return raw
    return float(raw)


def _fit_and_score(C: float, gamma, X_train: np.ndarray) -> SVDD:
    y_train = np.ones((X_train.shape[0], 1))  # 학습셋은 모두 정상(single class)으로 가정
    model = SVDD(C=C, kernel="rbf", gamma=gamma)
    model.fit(X_train, y_train)
    return model


def grid_search(X_train, X_valid, y_valid, C_list, gamma_list):
    best_f1, best_model, best_params = -1.0, None, {}
    for C in C_list:
        for gamma in gamma_list:
            model = _fit_and_score(C, gamma, X_train)
            pred = predictions_to_binary(model.predict(X_valid))
            f1 = f1_score(y_valid, pred, pos_label=1)
            print(f"[SVDD] C={C}, gamma={gamma} -> valid F1={f1:.4f}")
            if f1 > best_f1:
                best_f1, best_model, best_params = f1, model, {"C": C, "gamma": gamma, "f1": f1}
    print(f"[SVDD] best params: {best_params}")
    return best_model, best_params


def main():
    args = parse_args()
    set_seed(args.seed)

    train_df, valid_df, test_df, _ = load_and_scale(args.train_csv, args.valid_csv, args.test_csv)

    X_train = train_df.iloc[:, :-1].to_numpy()
    X_valid, y_valid = valid_df.iloc[:, :-1].to_numpy(), valid_df.iloc[:, -1].to_numpy().astype(int)
    X_test, y_test = test_df.iloc[:, :-1].to_numpy(), test_df.iloc[:, -1].to_numpy().astype(int)

    if args.search:
        model, best_params = grid_search(X_train, X_valid, y_valid, args.C_list, args.gamma_list)
    else:
        model = _fit_and_score(args.C, _parse_gamma(args.gamma), X_train)
        best_params = {"C": args.C, "gamma": args.gamma}

    valid_scores = model.get_distance(X_valid).ravel()
    threshold = get_threshold(valid_scores, y_valid)

    test_scores = model.get_distance(X_test).ravel()
    print(f"[SVDD] Test ROC AUC: {roc_auc(test_scores, y_test) * 100:.2f}")

    metrics = compute_metrics(test_scores, y_test, threshold)
    print("Test metrics:", metrics)
    print("사용된 조율 모수:", best_params)


if __name__ == "__main__":
    main()
