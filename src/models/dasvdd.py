"""DASVDD (Deep Autoencoding Support Vector Data Description) 이상탐지 모델.

DASVDD.ipynb 정리본. 1D 신호(길이 512)를 입력으로 받아 오토인코더로 복원하면서,
잠재벡터가 하나의 중심 c 주변에 모이도록(SVDD) 함께 학습한다.
이상 점수는 "복원오차 + gamma * SVDD 거리"로 계산한다.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.utils import roc_auc, weights_init_normal


class AutoEncoder(nn.Module):
    """1D Conv 기반 오토인코더. 입력 길이는 512로 고정되어 있다."""

    INPUT_LEN = 512

    def __init__(self, z_dim: int = 64):
        super().__init__()
        self.z_dim = z_dim

        self.conv1 = nn.Conv1d(1, 3, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(3, affine=False)
        self.conv2 = nn.Conv1d(3, 3, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(3, affine=False)
        self.fc1_1 = nn.Linear(self.INPUT_LEN * 3, 256, bias=False)
        self.bn3 = nn.BatchNorm1d(256, affine=False)
        self.fc1_2 = nn.Linear(256, 128, bias=False)
        self.bn4 = nn.BatchNorm1d(128, affine=False)
        self.fc1_3 = nn.Linear(128, z_dim, bias=False)
        self.bn5 = nn.BatchNorm1d(z_dim, affine=False)

        self.fc2_1 = nn.Linear(z_dim, 128, bias=False)
        self.bn6 = nn.BatchNorm1d(128, affine=False)
        self.fc2_2 = nn.Linear(128, 256, bias=False)
        self.bn7 = nn.BatchNorm1d(256, affine=False)
        self.fc2_3 = nn.Linear(256, self.INPUT_LEN * 3, bias=False)
        self.bn8 = nn.BatchNorm1d(self.INPUT_LEN * 3, affine=False)
        self.deconv1 = nn.ConvTranspose1d(3, 3, kernel_size=3, padding=1, bias=False)
        self.bn9 = nn.BatchNorm1d(3, affine=False)
        self.deconv2 = nn.ConvTranspose1d(3, 1, kernel_size=3, padding=1, bias=False)
        self.bn10 = nn.BatchNorm1d(1, affine=False)

    def encoder(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1, 1, self.INPUT_LEN)
        x = self.conv1(x)
        x = F.mish(self.bn1(x))
        x = self.conv2(x)
        x = F.mish(self.bn2(x))
        x = x.reshape(-1, 3 * self.INPUT_LEN)
        x = self.fc1_1(x)
        x = F.mish(self.bn3(x))
        x = self.fc1_2(x)
        x = F.mish(self.bn4(x))
        x = self.fc1_3(x)
        x = F.mish(self.bn5(x))
        return x

    def decoder(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc2_1(x)
        x = F.mish(self.bn6(x))
        x = self.fc2_2(x)
        x = F.mish(self.bn7(x))
        x = self.fc2_3(x)
        x = F.mish(self.bn8(x))
        x = x.reshape(-1, 3, self.INPUT_LEN)
        x = F.mish(self.bn9(self.deconv1(x)))
        x = F.mish(self.bn10(self.deconv2(x)))
        return x.reshape(-1, self.INPUT_LEN)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return z, x_hat


def tune_gamma(
    latent_dim: int,
    train_loader: DataLoader,
    device: torch.device,
    T: int = 20,
) -> tuple[float, torch.Tensor]:
    """복원오차(RE)와 SVDD 거리(R)의 비율로 gamma(두 loss의 균형 가중치)를 추정한다."""
    c = torch.randn(latent_dim).to(device)
    gamma = 0.0
    for _ in range(T):
        model = AutoEncoder(latent_dim).to(device)
        R, RE = 0.0, 0.0
        for x, _ in train_loader:
            x = x.float().to(device)
            z, x_hat = model(x)
            R += torch.mean(torch.sum((z - c) ** 2, dim=1))
            RE += torch.mean(torch.sum((x_hat - x) ** 2, dim=1))
        gamma += RE / R

    gamma = (gamma / T).detach().item()
    return gamma, c


class DASVDDTrainer:
    """오토인코더 복원오차 + SVDD 거리를 함께 최소화하도록 학습한다."""

    def __init__(self, args, train_loader: DataLoader, device: torch.device, gamma: float):
        self.args = args
        self.train_loader = train_loader
        self.device = device
        self.gamma = gamma

    def train(self) -> nn.Module:
        c = torch.randn(self.args.latent_dim).to(self.device)
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
            total_loss, ae_loss, svdd_loss = 0.0, 0.0, 0.0
            for x, _ in self.train_loader:
                x = x.float().to(self.device)

                optimizer.zero_grad()
                z, x_hat = net(x)
                R = torch.mean(torch.sum((z - c) ** 2, dim=1))
                loss = torch.mean(torch.sum((x_hat - x) ** 2, dim=1)) + self.gamma * R
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                ae_loss += torch.mean(torch.sum((x_hat - x) ** 2, dim=1)).item()
                svdd_loss += R.item()
            scheduler.step()

            if epoch % 10 == 0:
                n = len(self.train_loader)
                print(
                    f"[DASVDD] epoch {epoch} | total {total_loss/n:.6f} "
                    f"| ae {ae_loss/n:.6f} | svdd {svdd_loss/n:.6f}"
                )

        self.net = net
        self.c = c
        return net


def evaluate(
    net: nn.Module, c: torch.Tensor, gamma: float, dataloader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """복원오차 + gamma * SVDD 거리를 이상 점수로 사용해 평가한다."""
    net.eval()
    scores, labels = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.float().to(device)
            z, x_hat = net(x)
            score = torch.sum((x_hat - x) ** 2, dim=1) + gamma * torch.sum((z - c) ** 2, dim=1)
            scores.append(score.detach().cpu())
            labels.append(y.cpu())

    labels = torch.cat(labels).numpy()
    scores = torch.cat(scores).numpy()
    print(f"[DASVDD] ROC AUC: {roc_auc(scores, labels) * 100:.2f}")
    return labels, scores
