"""공통 유틸리티 함수 모음.

- 디바이스 선택 (GPU가 없거나 지정한 GPU 번호가 없을 때 안전하게 fallback)
- 랜덤 시드 고정
- 가중치 초기화
- 평가지표 계산 및 threshold 산출
"""
from __future__ import annotations

import pickle
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def get_device(gpu: int | None = None) -> torch.device:
    """사용 가능한 디바이스를 반환한다.

    gpu 번호가 실제 존재하는 GPU 개수를 넘어서면(예: GPU가 1개인데 gpu=3)
    'invalid device ordinal' 런타임 에러가 나므로, 여기서 미리 검증해서
    안전하게 cuda:0 혹은 cpu로 fallback 한다.
    """
    if gpu is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available() and gpu < torch.cuda.device_count():
        return torch.device(f"cuda:{gpu}")

    if torch.cuda.is_available():
        print(
            f"[경고] cuda:{gpu} 는 존재하지 않습니다 "
            f"(사용 가능한 GPU {torch.cuda.device_count()}개). cuda:0으로 대체합니다."
        )
        return torch.device("cuda:0")

    print("[경고] GPU를 사용할 수 없어 CPU로 실행합니다.")
    return torch.device("cpu")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def weights_init_normal(m: nn.Module) -> None:
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 and classname != "Conv":
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("Linear") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)


def get_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """validation score 분포에서, 실제 이상치 개수만큼을 상위 threshold로 잡는다."""
    n_abnormal = int(sum(labels == 1))
    return sorted(scores, reverse=True)[n_abnormal]


def compute_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    """score를 threshold로 이진화한 뒤 accuracy/precision/recall/f1을 계산한다."""
    pred = scores.copy()
    pred[pred < threshold] = 0
    pred[pred >= threshold] = 1
    return {
        "accuracy": accuracy_score(pred, labels),
        "precision": precision_score(pred, labels),
        "recall": recall_score(pred, labels),
        "f1": f1_score(pred, labels, pos_label=1),
    }


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    return float(roc_auc_score(labels, scores))


def save_pickle(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str | Path):
    with open(path, "rb") as f:
        return pickle.load(f)
