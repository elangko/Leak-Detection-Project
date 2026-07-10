"""AnoGAN 이상탐지 모델.

AnoGAN__1_.ipynb 정리본. GAN을 정상 데이터로만 학습시킨 뒤, 테스트 시점에는
generator의 출력이 실제 샘플과 가장 비슷해지도록 latent vector z를 최적화한다.
이 최적화 과정에서의 residual loss(복원오차) + discrimination loss(특성 차이)를
이상 점수로 사용한다.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.utils import roc_auc, weights_init_normal


class Generator(nn.Module):
    def __init__(self, dim: int, z_dim: int = 64):
        super().__init__()
        self.z_dim = z_dim
        self.dim = dim

        self.fc1 = nn.Linear(z_dim, 128, bias=True)
        self.bn1 = nn.BatchNorm1d(128, affine=True)
        self.fc2 = nn.Linear(128, 256, bias=True)
        self.bn2 = nn.BatchNorm1d(256, affine=True)
        self.fc3 = nn.Linear(256, dim * 3, bias=True)
        self.bn3 = nn.BatchNorm1d(dim * 3, affine=True)
        self.deconv1 = nn.ConvTranspose1d(3, 3, kernel_size=3, padding=1, bias=True)
        self.bn4 = nn.BatchNorm1d(3, affine=True)
        self.deconv2 = nn.ConvTranspose1d(3, 1, kernel_size=3, padding=1, bias=True)
        self.bn5 = nn.BatchNorm1d(1, affine=True)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.fc1(z)
        z = F.mish(self.bn1(z))
        z = self.fc2(z)
        z = F.mish(self.bn2(z))
        z = self.fc3(z)
        z = F.mish(self.bn3(z))
        z = z.reshape(-1, 3, self.dim)

        z = F.mish(self.bn4(self.deconv1(z)))
        z = self.bn5(self.deconv2(z))
        return z.reshape(-1, self.dim)


class Discriminator(nn.Module):
    def __init__(self, dim: int, z_dim: int = 64):
        super().__init__()
        self.z_dim = z_dim
        self.dim = dim

        self.conv1 = nn.Conv1d(1, 3, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm1d(3, affine=True)
        self.conv2 = nn.Conv1d(3, 3, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm1d(3, affine=True)
        self.fc1 = nn.Linear(dim * 3, 256, bias=True)
        self.bn3 = nn.BatchNorm1d(256, affine=True)
        self.fc2 = nn.Linear(256, 128, bias=True)
        self.bn4 = nn.BatchNorm1d(128, affine=True)
        self.fc3 = nn.Linear(128, z_dim, bias=True)
        self.bn5 = nn.BatchNorm1d(z_dim, affine=True)
        self.fc4 = nn.Linear(z_dim, 1, bias=True)
        self.bn6 = nn.BatchNorm1d(1, affine=True)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1, 1, self.dim)
        x = self.conv1(x)
        x = F.mish(self.bn1(x))
        x = self.conv2(x)
        x = F.mish(self.bn2(x))
        x = x.reshape(-1, 3 * self.dim)
        x = self.fc1(x)
        x = F.mish(self.bn3(x))
        x = self.fc2(x)
        x = F.mish(self.bn4(x))
        x = self.fc3(x)
        return F.mish(self.bn5(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        return torch.sigmoid(self.bn6(self.fc4(features)))


class GANTrainer:
    """정상 데이터만으로 Generator/Discriminator를 적대적으로 학습한다."""

    def __init__(self, args, train_loader: DataLoader, device: torch.device):
        self.args = args
        self.train_loader = train_loader
        self.device = device

    def train(self) -> tuple[Discriminator, Generator, list, list]:
        G = Generator(self.args.dim, self.args.z_dim).to(self.device)
        D = Discriminator(self.args.dim, self.args.z_dim).to(self.device)
        criterion = nn.BCELoss()

        optimizer_G = torch.optim.Adam(G.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        optimizer_D = torch.optim.Adam(D.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

        D.apply(weights_init_normal)
        G.apply(weights_init_normal)
        D.train()
        G.train()

        d_losses, g_losses = [], []

        for epoch in range(self.args.epochs):
            for x, _ in self.train_loader:
                x = x.float().to(self.device)

                # --- Discriminator ---
                optimizer_D.zero_grad()
                real_output = D(x)
                real_label = torch.ones_like(real_output, device=self.device)

                z = torch.randn(self.args.batch_size, self.args.z_dim, device=self.device)
                fake_data = G(z)
                fake_output = D(fake_data.detach())
                fake_label = torch.zeros_like(fake_output, device=self.device)

                d_loss = criterion(real_output, real_label) + criterion(fake_output, fake_label)
                d_loss.backward()
                optimizer_D.step()

                p_real = real_output.mean().item()
                p_fake = fake_output.mean().item()

                # --- Generator (2x updates per D update, as in original) ---
                for _ in range(2):
                    optimizer_G.zero_grad()
                    fake_data = G(z)
                    fake_output = D(fake_data)
                    g_loss = criterion(fake_output, torch.ones_like(fake_output, device=self.device))
                    g_loss.backward()
                    optimizer_G.step()

                d_losses.append(d_loss.item())
                g_losses.append(g_loss.item())

            if epoch % 50 == 0:
                print(
                    f"[AnoGAN] epoch {epoch} | D {d_loss.item():.4f} | G {g_loss.item():.4f} "
                    f"| P(real) {p_real:.4f} | P(fake) {p_fake:.4f}"
                )

        self.D, self.G = D, G
        self.d_losses, self.g_losses = d_losses, g_losses
        return D, G, d_losses, g_losses


def residual_loss(real: torch.Tensor, generated: torch.Tensor) -> torch.Tensor:
    return torch.sum(torch.abs(real - generated), axis=1)


def discriminator_loss(D: Discriminator, real: torch.Tensor, generated: torch.Tensor) -> torch.Tensor:
    real_features = D.forward_features(real)
    generated_features = D.forward_features(generated)
    return torch.sum(torch.abs(real_features - generated_features), axis=1)


def anomaly_loss(res_loss: torch.Tensor, disc_loss: torch.Tensor, l: float = 0.1) -> torch.Tensor:
    return (1 - l) * res_loss + l * disc_loss


def estimate_anomaly_score(
    real: torch.Tensor, generated: torch.Tensor, D: Discriminator, l: float = 0.5
) -> np.ndarray:
    res_loss = residual_loss(real, generated)
    disc_loss = discriminator_loss(D, real, generated)
    return anomaly_loss(res_loss, disc_loss, l=l).cpu().data.numpy()


def optimize_latent(
    args, G: Generator, D: Discriminator, dataloader: DataLoader, device: torch.device, steps: int = 200
) -> list[np.ndarray]:
    """각 배치마다 generator 출력이 실제 데이터와 가장 비슷해지도록 z를 최적화한다."""
    G.eval()
    D.eval()
    latent_space = []

    for x, _ in dataloader:
        real_data = x.float().to(device)
        z = torch.randn(len(x), args.z_dim, device=device, requires_grad=True)
        optimizer_z = torch.optim.Adam([z], lr=0.1)

        for step in range(steps + 1):
            generated_data = G(z)
            optimizer_z.zero_grad()
            loss = anomaly_loss(
                residual_loss(real_data, generated_data),
                discriminator_loss(D, real_data, generated_data),
                l=0.1,
            ).mean()
            loss.backward()
            optimizer_z.step()

            if step == steps:
                latent_space.append(z.cpu().data.numpy())

    return latent_space


def evaluate(
    G: Generator, D: Discriminator, latent_space: list[np.ndarray], dataloader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    G.eval()
    D.eval()
    scores, labels = np.array([]), np.array([])

    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            real_data = x.float().to(device)
            z = torch.as_tensor(latent_space[i], device=device, dtype=torch.float32)
            generated_data = G(z).to(device)
            scores = np.append(scores, estimate_anomaly_score(real_data, generated_data, D))
            labels = np.append(labels, y.cpu())

    print(f"[AnoGAN] ROC AUC: {roc_auc(scores, labels) * 100:.2f}")
    return labels, scores
