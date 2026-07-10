"""One-Class SVM 학습/평가 CLI.

기본 사용법:
    python -m src.train_ocsvm \
        --train-csv ./data/all_train.csv \
        --valid-csv ./data/valid.csv \
        --test-csv ./data/all_test.csv

딥러닝 모형들과 달리 OC-SVM은 (nu, gamma) grid search로 조율 모수를 선택한다.
"""
from __future__ import annotations

import argparse

from src.data.dataset import load_and_scale
from src.models.one_class_svm import anomaly_score, grid_search, predict_labels
from src.utils import compute_metrics, get_threshold, roc_auc, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="One-Class SVM 학습/평가")
    p.add_argument("--train-csv", type=str, default="./data/all_train.csv")
    p.add_argument("--valid-csv", type=str, default="./data/valid.csv")
    p.add_argument("--test-csv", type=str, default="./data/all_test.csv")
    p.add_argument("--nu", type=float, nargs="+", default=[0.01, 0.05, 0.1, 0.2])
    p.add_argument("--gamma", type=float, nargs="+", default=[1e-4, 1e-5, 1e-6, 1e-7])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    train_df, valid_df, test_df, _ = load_and_scale(args.train_csv, args.valid_csv, args.test_csv)

    X_train = train_df.iloc[:, :-1].to_numpy()
    X_valid, y_valid = valid_df.iloc[:, :-1].to_numpy(), valid_df.iloc[:, -1].to_numpy().astype(int)
    X_test, y_test = test_df.iloc[:, :-1].to_numpy(), test_df.iloc[:, -1].to_numpy().astype(int)

    model, best_params = grid_search(X_train, X_valid, y_valid, nu_list=args.nu, gamma_list=args.gamma)

    valid_scores = anomaly_score(model, X_valid)
    threshold = get_threshold(valid_scores, y_valid)

    test_scores = anomaly_score(model, X_test)
    print(f"[OC-SVM] Test ROC AUC: {roc_auc(test_scores, y_test) * 100:.2f}")

    metrics = compute_metrics(test_scores, y_test, threshold)
    print("Test metrics:", metrics)
    print("선택된 조율 모수:", best_params)


if __name__ == "__main__":
    main()
