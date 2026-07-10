"""Deep SVDD 이상탐지 모델.

Deep_SVDD.ipynb 정리본. 논문 2.5절 Deep SVDD 모형에 해당한다.

두 단계로 학습한다.
1. 사전학습(pretrain): 오토인코더를 복원오차로 학습해 인코더 가중치를 얻는다.
2. 본학습(train): 사전학습된 인코더 가중치로 Deep SVDD 네트워크를 초기화하고,
   초구의 중점 c(사전학습된 잠재 표현의 평균)에 가까워지도록 학습한다.

원본 노트북은 사전학습 가중치를 디스크(`./weights/pretrained_parameters.pth`)에 저장한 뒤
다시 불러오는 방식이었으나, 여기서는 메모리 상에서 바로 전달하도록 정리했다.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.utils import roc_auc, weights_init_normal


class DeepSVDDNetwork(nn.Module):
    """잠재 표현만 출력하는 인코더 (디코더 없음)."""

    def __init__(self, dim: int, z_dim: int = 64):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim

        self.conv1 = nn.Conv1d(1, 3, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(3, affine=False)
        self.conv2 = nn.Conv1d(3, 3, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(3, affine=False)
        self.fc1_1 = nn.Linear(dim * 3, 256, bias=False)
        self.fc1_2 = nn.Linear(256, 128, bias=False)
        self.fc1_3 = nn.Linear(128, z_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1, 1, self.dim)
        x = self.conv1(x)
        x = F.mish(self.bn1(x))
        x = self.conv2(x)
        x = F.mish(self.bn2(x))
        x = x.reshape(-1, 3 * self.dim)
        x = F.mish(self.fc1_1(x))
        x = F.mish(self.fc1_2(x))
        return self.fc1_3(x)


class PretrainAutoEncoder(nn.Module):
    """Deep SVDD 인코더와 동일한 구조 + 대칭 디코더. 사전학습 전용."""

    def __init__(self, dim: int, z_dim: int = 64):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim

        self.conv1 = nn.Conv1d(1, 3, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(3, affine=False)
        self.conv2 = nn.Conv1d(3, 3, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(3, affine=False)
        self.fc1_1 = nn.Linear(dim * 3, 256, bias=False)
        self.fc1_2 = nn.Linear(256, 128, bias=False)
        self.fc1_3 = nn.Linear(128, z_dim, bias=False)

        self.fc2_1 = nn.Linear(z_dim, 128, bias=False)
        self.fc2_2 = nn.Linear(128, 256, bias=False)
        self.fc2_3 = nn.Linear(256, dim * 3, bias=False)
        self.deconv1 = nn.ConvTranspose1d(3, 3, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm1d(3, affine=False)
        self.deconv2 = nn.ConvTranspose1d(3, 1, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm1d(1, affine=False)

    def encoder(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1, 1, self.dim)
        x = self.conv1(x)
        x = F.mish(self.bn1(x))
        x = self.conv2(x)
        x = F.mish(self.bn2(x))
        x = x.reshape(-1, 3 * self.dim)
        x = F.mish(self.fc1_1(x))
        x = F.mish(self.fc1_2(x))
        return self.fc1_3(x)

    def decoder(self, x: torch.Tensor) -> torch.Tensor:
        x = F.mish(self.fc2_1(x))
        x = F.mish(self.fc2_2(x))
        x = F.mish(self.fc2_3(x))
        x = x.reshape(-1, 3, self.dim)
        x = F.mish(self.bn3(self.deconv1(x)))
        x = self.bn4(self.deconv2(x))
        return x.reshape(-1, self.dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class DeepSVDDTrainer:
    def __init__(self, args, train_loader: DataLoader, device: torch.device):
        self.args = args
        self.train_loader = train_loader
        self.device = device

    def pretrain(self) -> tuple[dict, torch.Tensor]:
        """오토인코더를 복원오차로 사전학습하고, 인코더 state_dict와 초구 중점 c를 반환한다."""
        ae = PretrainAutoEncoder(dim=self.args.dim, z_dim=self.args.latent_dim).to(self.device)
        optimizer = torch.optim.Adam(
            ae.parameters(), lr=self.args.lr_ae, weight_decay=self.args.weight_decay_ae
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.args.lr_milestones, gamma=0.1
        )
        ae.apply(weights_init_normal)
        ae.train()

        for epoch in range(self.args.num_epochs_ae):
            total_loss = 0.0
            for x, _ in self.train_loader:
                x = x.float().to(self.device)
                optimizer.zero_grad()
                x_hat = ae(x)
                loss = torch.mean(torch.sum((x_hat - x) ** 2, dim=1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()

            if epoch % 10 == 0:
                print(f"[Deep SVDD 사전학습] epoch {epoch} | recon loss {total_loss/len(self.train_loader):.6f}")

        c = self._init_center(ae)
        return ae.state_dict(), c

    def _init_center(self, ae: nn.Module, eps: float = 0.1) -> torch.Tensor:
        """사전학습된 인코더로 잠재 표현의 평균을 구해 초구의 중점으로 사용한다."""
        ae.eval()
        z_list = []
        with torch.no_grad():
            for x, _ in self.train_loader:
                x = x.float().to(self.device)
                z_list.append(ae.encoder(x).detach())
        z = torch.cat(z_list)
        c = torch.mean(z, dim=0)
        # 중점이 0에 너무 가까우면 사소한 특성으로 trivial solution에 빠질 수 있어 clamp한다.
        c[(c.abs() < eps) & (c < 0)] = -eps
        c[(c.abs() < eps) & (c > 0)] = eps
        return c

    def train(self, pretrained_state: dict | None = None, c: torch.Tensor | None = None) -> tuple[nn.Module, torch.Tensor]:
        net = DeepSVDDNetwork(dim=self.args.dim, z_dim=self.args.latent_dim).to(self.device)

        if pretrained_state is not None:
            net.load_state_dict(pretrained_state, strict=False)
        else:
            net.apply(weights_init_normal)
        if c is None:
            c = torch.randn(self.args.latent_dim).to(self.device)

        optimizer = torch.optim.Adam(
            net.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.args.lr_milestones, gamma=0.1
        )

        net.train()
        for epoch in range(self.args.num_epochs):
            total_loss = 0.0
            for x, _ in self.train_loader:
                x = x.float().to(self.device)

                optimizer.zero_grad()
                z = net(x)
                loss = torch.mean(torch.sum((z - c) ** 2, dim=1))
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            scheduler.step()

            if epoch % 10 == 0:
                print(f"[Deep SVDD] epoch {epoch} | loss {total_loss/len(self.train_loader):.6f}")

        self.net = net
        self.c = c
        return net, c


def evaluate(
    net: nn.Module, c: torch.Tensor, dataloader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """중점 c까지의 거리를 이상 점수로 사용해 평가한다."""
    net.eval()
    scores, labels = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.float().to(device)
            z = net(x)
            score = torch.sum((z - c) ** 2, dim=1)
            scores.append(score.detach().cpu())
            labels.append(y.cpu())

    labels = torch.cat(labels).numpy()
    scores = torch.cat(scores).numpy()
    print(f"[Deep SVDD] ROC AUC: {roc_auc(scores, labels) * 100:.2f}")
    return labels, scores
