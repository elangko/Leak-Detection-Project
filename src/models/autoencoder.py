"""Convolutional Autoencoder(CAE) 이상탐지 모델.

CAE.ipynb 정리본. DASVDD와 동일한 1D Conv 오토인코더 구조를 사용하지만,
SVDD 항 없이 순수 복원오차(reconstruction error)만으로 학습한다.
이상 점수는 복원오차 하나로만 계산한다 (DASVDD.evaluate에서 gamma=0인 경우와 동일).

논문 2.3절 Autoencoder 모형에 해당한다.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.dasvdd import AutoEncoder
from src.utils import roc_auc, weights_init_normal

__all__ = ["AutoEncoder", "CAETrainer", "evaluate"]


class CAETrainer:
    """복원오차만을 최소화하도록 오토인코더를 학습한다."""

    def __init__(self, args, train_loader: DataLoader, device: torch.device):
        self.args = args
        self.train_loader = train_loader
        self.device = device

    def train(self) -> nn.Module:
        net = AutoEncoder(self.args.latent_dim).to(self.device)
        optimizer = torch.optim.Adam(
            net.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.args.lr_milestones, gamma=0.1
        )
        net.apply(weights_init_normal)
        net.train()

        for epoch in range(self.args.num_epochs):
            total_loss = 0.0
            for x, _ in self.train_loader:
                x = x.float().to(self.device)

                optimizer.zero_grad()
                _, x_hat = net(x)
                loss = torch.mean(torch.sum((x_hat - x) ** 2, dim=1))
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            scheduler.step()

            if epoch % 10 == 0:
                print(f"[CAE] epoch {epoch} | recon loss {total_loss/len(self.train_loader):.6f}")

        self.net = net
        return net


def evaluate(net: nn.Module, dataloader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """복원오차를 이상 점수로 사용해 평가한다."""
    net.eval()
    scores, labels = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.float().to(device)
            _, x_hat = net(x)
            score = torch.sum((x_hat - x) ** 2, dim=1)
            scores.append(score.detach().cpu())
            labels.append(y.cpu())

    labels = torch.cat(labels).numpy()
    scores = torch.cat(scores).numpy()
    print(f"[CAE] ROC AUC: {roc_auc(scores, labels) * 100:.2f}")
    return labels, scores
