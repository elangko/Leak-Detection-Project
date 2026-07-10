"""DASVDD 학습/평가 CLI.

기본 사용법:
    python -m src.train_dasvdd \
        --train-csv ./data/all_train.csv \
        --valid-csv ./data/valid.csv \
        --test-csv ./data/all_test.csv

시나리오 실험(다양한 오염/이상비율 조건)을 반복 실행하려면 --scenario 1 또는 2를 사용한다.
"""
from __future__ import annotations

import argparse
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data.dataset import get_dataloaders, load_and_scale
from src.models.dasvdd import DASVDDTrainer, evaluate, tune_gamma
from src.utils import compute_metrics, get_device, get_threshold, save_pickle, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="DASVDD 학습/평가")
    p.add_argument("--train-csv", type=str, default="./data/all_train.csv")
    p.add_argument("--valid-csv", type=str, default="./data/valid.csv")
    p.add_argument("--test-csv", type=str, default="./data/all_test.csv")
    p.add_argument("--data-dir", type=str, default="./data", help="시나리오 실험용 원본 csv들이 위치한 폴더")

    p.add_argument("--num-epochs", type=int, default=101)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=5e-6)
    p.add_argument("--lr-milestones", type=int, nargs="+", default=[1000])
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--normal-class", type=int, default=0)
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--scenario", type=int, choices=[1, 2], default=None,
                    help="1: 테스트 이상비율 변화 실험, 2: 학습데이터 오염비율 변화 실험")
    p.add_argument("--n-repeat", type=int, default=10, help="시나리오 실험 반복 횟수")
    p.add_argument("--proportions", type=float, nargs="+", default=[0.01, 0.05, 0.1, 0.2, 0.3])
    p.add_argument("--output", type=str, default="./results/dasvdd_result.pkl")
    return p.parse_args()


def run_train_eval(args, device) -> None:
    """단일 train/valid/test 셋으로 한 번 학습하고 평가한다."""
    train_df, valid_df, test_df, _ = load_and_scale(args.train_csv, args.valid_csv, args.test_csv)
    dataloader_train, dataloader_valid, dataloader_test = get_dataloaders(
        train_df, valid_df, test_df, args.batch_size
    )

    gamma, c = tune_gamma(args.latent_dim, dataloader_train, device)
    print(f"gamma = {gamma:.6f}")

    trainer = DASVDDTrainer(args, dataloader_train, device, gamma)
    net = trainer.train()

    labels, scores = evaluate(net, c, gamma, dataloader_valid, device)
    threshold = get_threshold(scores, labels)

    labels, scores = evaluate(net, c, gamma, dataloader_test, device)
    metrics = compute_metrics(scores, labels, threshold)
    print("Test metrics:", metrics)


def _run_once(args, device, col_list, tmp_1, tmp_2, tmp_3):
    dataloader_train, dataloader_valid, dataloader_test = get_dataloaders(
        tmp_1, tmp_2, tmp_3, args.batch_size
    )
    start = time.time()

    gamma, c = tune_gamma(args.latent_dim, dataloader_train, device)
    trainer = DASVDDTrainer(args, dataloader_train, device, gamma)
    net = trainer.train()

    labels, scores = evaluate(net, c, gamma, dataloader_valid, device)
    threshold = get_threshold(scores, labels)
    elapsed = time.time() - start

    labels, scores = evaluate(net, c, gamma, dataloader_test, device)
    metrics = compute_metrics(scores, labels, threshold)
    return elapsed, metrics


def scenario_1(args, device, col_list) -> dict:
    """테스트 데이터의 이상치 비율을 바꿔가며 성능을 측정한다."""
    data_dir = args.data_dir
    normal_train = pd.read_csv(f"{data_dir}/normal-training.csv")
    in_train = pd.read_csv(f"{data_dir}/in-training.csv")
    normal_test = pd.read_csv(f"{data_dir}/normal_test.csv")
    in_test = pd.read_csv(f"{data_dir}/in_test.csv")
    valid = pd.read_csv(f"{data_dir}/valid.csv")

    tmp_1 = normal_train.iloc[:, col_list].copy()
    tmp_1.loc[tmp_1["leaktype"] == "normal", "leaktype"] = 0
    tmp_1 = tmp_1.astype({"leaktype": "int64"})

    tmp_2 = valid.copy()
    tmp_2.columns = tmp_1.columns

    scaler = StandardScaler()
    scaler.fit(tmp_1)
    tmp_1 = pd.DataFrame(scaler.transform(tmp_1))
    tmp_2 = pd.DataFrame(scaler.transform(tmp_2))

    results = {"cal_time": [], "accuracy": [], "f1": [], "precision": [], "recall": []}

    for p in args.proportions:
        print(f"[scenario_1] proportion={p}")
        tmp_3 = pd.concat([normal_test, in_test.sample(int(len(normal_test) * p))], axis=0)
        tmp_3 = tmp_3.iloc[:, col_list].copy().sample(frac=1).reset_index(drop=True)
        tmp_3.loc[tmp_3["leaktype"] == "normal", "leaktype"] = 0
        tmp_3.loc[tmp_3["leaktype"] == "in", "leaktype"] = 1
        tmp_3 = tmp_3.astype({"leaktype": "int64"})
        tmp_3 = pd.DataFrame(scaler.transform(tmp_3))

        elapsed, metrics = _run_once(args, device, col_list, tmp_1, tmp_2, tmp_3)
        results["cal_time"].append(elapsed)
        for k in ("accuracy", "f1", "precision", "recall"):
            results[k].append(metrics[k])

    return results


def scenario_2(args, device, col_list) -> dict:
    """학습 데이터에 섞인 이상치(오염) 비율을 바꿔가며 성능을 측정한다."""
    data_dir = args.data_dir
    normal_train = pd.read_csv(f"{data_dir}/normal-training.csv")
    in_train = pd.read_csv(f"{data_dir}/in-training.csv")
    normal_test = pd.read_csv(f"{data_dir}/normal_test.csv")
    in_test = pd.read_csv(f"{data_dir}/in_test.csv")
    valid = pd.read_csv(f"{data_dir}/valid.csv")

    tmp_3 = pd.concat([normal_test, in_test.sample(int(len(normal_test) * 0.1))], axis=0)
    tmp_3 = tmp_3.sample(frac=1).reset_index(drop=True).iloc[:, col_list].copy()
    tmp_3.loc[tmp_3["leaktype"] == "normal", "leaktype"] = 0
    tmp_3.loc[tmp_3["leaktype"] == "in", "leaktype"] = 1
    tmp_3 = tmp_3.astype({"leaktype": "int64"})

    tmp_2 = valid.copy()
    tmp_2.columns = tmp_3.columns

    results = {"cal_time": [], "accuracy": [], "f1": [], "precision": [], "recall": []}

    for p in args.proportions:
        print(f"[scenario_2] contamination={p}")
        tmp_1 = pd.concat([normal_train, in_train.sample(int(len(normal_train) * p))], axis=0)
        tmp_1 = tmp_1.sample(frac=1).reset_index(drop=True).iloc[:, col_list].copy()
        tmp_1.loc[:, "leaktype"] = 0
        tmp_1 = tmp_1.astype({"leaktype": "int64"})

        scaler = StandardScaler()
        scaler.fit(tmp_1)
        tmp_1_s = pd.DataFrame(scaler.transform(tmp_1))
        tmp_2_s = pd.DataFrame(scaler.transform(tmp_2))
        tmp_3_s = pd.DataFrame(scaler.transform(tmp_3))

        elapsed, metrics = _run_once(args, device, col_list, tmp_1_s, tmp_2_s, tmp_3_s)
        results["cal_time"].append(elapsed)
        for k in ("accuracy", "f1", "precision", "recall"):
            results[k].append(metrics[k])

    return results


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.gpu)

    if args.scenario is None:
        run_train_eval(args, device)
        return

    # 원본 노트북 기준 컬럼 인덱스 (leaktype 포함, id/site 등 메타 컬럼 제외)
    col_list = [i for i in range(7, 519)] + [5]

    scenario_fn = scenario_1 if args.scenario == 1 else scenario_2
    all_runs = []
    for i in range(args.n_repeat):
        print(f"=== repeat {i + 1}/{args.n_repeat} ===")
        result = scenario_fn(args, device, col_list)
        row = np.concatenate(
            [result["cal_time"], result["accuracy"], result["f1"], result["precision"], result["recall"]]
        )
        all_runs.append(row)

    save_pickle(all_runs, args.output)
    print(f"결과 저장 완료: {args.output}")


if __name__ == "__main__":
    main()
