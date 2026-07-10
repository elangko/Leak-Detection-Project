"""Support Vector Data Description(SVDD) 이상탐지 모델.

svdd.ipynb 정리본. 논문 2.2절 SVDD 모형에 해당한다.
scikit-learn과 호환되는 인터페이스(fit/predict/decision_function)를 제공하며,
`cvxopt`의 QP solver로 라그랑주 듀얼 문제를 풀어 초구(hypersphere)를 학습한다.

원본 노트북에는 2D 시각화용 plot_boundary / plot_distance 메서드가 포함돼 있었으나,
이 프로젝트의 입력은 512차원 주파수 스펙트럼이라 사용할 수 없어 정리 과정에서 제외했다.
"""
from __future__ import annotations

import time
import warnings
from collections import defaultdict

import numpy as np
from cvxopt import matrix, solvers
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import pairwise_kernels


class SVDD(BaseEstimator, OutlierMixin):
    """커널 기반 Support Vector Data Description.

    Parameters
    ----------
    C : float, default=0.9
        정규화 강도의 역수. 값이 작을수록 규제가 강해진다.
    kernel : {'linear', 'poly', 'rbf', 'sigmoid'}, default='rbf'
    gamma : {'scale', 'auto'} or float, default='scale'
    """

    def __init__(
        self,
        C: float = 0.9,
        kernel: str = "rbf",
        degree: int = 3,
        gamma=None,
        coef0: float = 1,
        verbose: bool = False,
        n_jobs: int | None = None,
    ):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.alpha_tolerance = 1e-6

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def fit(self, X: np.ndarray, y: np.ndarray | None = None, weight: np.ndarray | None = None) -> "SVDD":
        start = time.time()
        self.X, self.y, self.y_type, self.exist_y = self._check_X_y(X, y)

        if self.y_type == "single":
            self.C_ = [self.C, 1]
        else:  # hybrid (1과 -1이 섞인 경우)
            self.C_ = [self.C, 2 / self.n_negative_samples]

        self.weight = weight if weight is not None else np.ones((self.n_samples, 1))
        self._resolve_gamma(X)

        K = self._kernel(self.X, self.X)
        self._solve(K)

        self.predicted_y_ = self.predict(self.X, self.y)
        self.accuracy_ = accuracy_score(self.y, self.predicted_y_)
        self.running_time_ = time.time() - start

        if self.verbose:
            print(
                f"[SVDD] fit 완료 | time={self.running_time_:.3f}s | "
                f"SVs={self.n_support_vectors_} ({100*self.n_support_vectors_ratio_:.2f}%) | "
                f"train acc={100*self.accuracy_:.2f}%"
            )
        return self

    def predict(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        X, y, _, exist_y = self._check_X_y(X, y)
        distance = self.get_distance(X)
        pred = np.ones(X.shape[0])
        pred[(distance > self.radius_).ravel()] = -1

        if self.verbose and exist_y:
            print(f"[SVDD] predict acc={100*accuracy_score(y, pred.reshape(-1, 1)):.2f}%")
        return pred

    def fit_predict(self, X, y=None, weight=None) -> np.ndarray:
        self.fit(X, y, weight)
        return self.predict(X, y)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """양수면 정상(inlier), 음수면 이상치(outlier)에 가깝다."""
        return self.radius_ - self.get_distance(X)

    def get_distance(self, X: np.ndarray) -> np.ndarray:
        """학습된 초구 중점으로부터 각 샘플까지의 거리."""
        K = self._kernel(X, self.X)
        K_self = self._kernel(X, X)
        weighted = np.dot(np.ones((X.shape[0], 1)), self.alpha_.T) * K
        cross_term = -2 * np.sum(weighted, axis=1, keepdims=True)
        dist_sq = np.diag(K_self).reshape(-1, 1) + self.offset_ + cross_term
        return np.sqrt(np.maximum(dist_sq, 0))

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    @property
    def n_samples(self):
        return self.X.shape[0]

    @property
    def n_negative_samples(self):
        return int(np.sum(self.y == -1))

    def _resolve_gamma(self, X: np.ndarray) -> None:
        if self.gamma == 0:
            raise ValueError("gamma=0은 허용되지 않습니다. 'auto'를 사용하세요.")
        if self.gamma is None:
            self.gamma = "scale"
        if isinstance(self.gamma, str):
            if self.gamma == "scale":
                var = X.var()
                self.gamma = 1.0 / (X.shape[1] * var) if var != 0 else 1.0
            elif self.gamma == "auto":
                self.gamma = 1.0 / X.shape[1]
            else:
                raise ValueError("gamma는 'scale', 'auto' 또는 float 이어야 합니다.")

    def _kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return pairwise_kernels(
            X, Y, metric=self.kernel, filter_params=True, n_jobs=self.n_jobs,
            gamma=self.gamma, degree=self.degree, coef0=self.coef0,
        )

    def _solve(self, K: np.ndarray) -> None:
        """라그랑주 듀얼 문제를 cvxopt QP solver로 푼다."""
        solvers.options["show_progress"] = False
        K_signed = np.multiply(self.y * self.y.T, K)

        n = K_signed.shape[0]
        P = matrix(K_signed + K_signed.T)
        q = matrix(-np.multiply(self.y, np.diagonal(K_signed).reshape(-1, 1)))

        G = matrix(np.append(-np.eye(n), np.eye(n), axis=0))
        h_upper = np.ones([n, 1])
        if self.y_type == "single":
            h_upper[self.y == 1] = self.C_[0] * self.weight[self.y == 1]
        else:
            h_upper[self.y == 1] = self.C_[0] * self.weight[self.y == 1]
            h_upper[self.y == -1] = self.C_[1] * self.weight[self.y == -1]
        h = matrix(np.append(np.zeros([n, 1]), h_upper, axis=0))

        A = matrix(np.ones([n, 1]).T)
        b = matrix(np.ones([1, 1]))

        sol = solvers.qp(P, q, G, h, A, b)
        alpha = np.array(sol["x"]) if len(np.array(sol["x"])) else self._fallback_alpha(n)
        alpha = self.y * alpha

        sv_idx = np.where(np.abs(alpha) > self.alpha_tolerance)[0]
        alpha[np.abs(alpha) < self.alpha_tolerance] = 0

        tmp_alpha = alpha[sv_idx, 0]
        tmp_bound = h_upper[sv_idx, 0]
        boundary_idx = sv_idx[
            np.array(list(
                set(np.where(tmp_alpha < tmp_bound)[0]) & set(np.where(tmp_alpha > self.alpha_tolerance)[0])
            ))
        ]

        self.alpha_ = alpha
        self.support_vector_indices_ = sv_idx
        self.support_vectors_ = self.X[sv_idx, :]
        self.n_support_vectors_ = sv_idx.shape[0]
        self.n_support_vectors_ratio_ = self.n_support_vectors_ / self.n_samples
        if self.n_support_vectors_ratio_ > 0.5:
            warnings.warn("Support vector 비율이 50%를 초과합니다 - 과적합 가능성이 있습니다.")

        weighted = np.dot(np.ones((self.n_samples, 1)), alpha.T) * K
        cross_term = -2 * np.sum(weighted, axis=1, keepdims=True)
        self.offset_ = float(np.sum(np.multiply(np.dot(alpha, alpha.T), K)))
        self.radius_ = float(
            np.sqrt(np.mean(np.diag(K)) + self.offset_ + np.mean(cross_term[boundary_idx, 0]))
        )

    def _fallback_alpha(self, n: int) -> np.ndarray:
        warnings.warn("SVDD 최적화 해를 찾지 못했습니다. fallback alpha를 사용합니다.")
        alpha = np.zeros((n, 1))
        alpha[0][0] = 1
        return alpha

    def _check_X_y(self, X: np.ndarray, y: np.ndarray | None):
        if y is None:
            y = np.ones((X.shape[0], 1))
            exist_y = False
        else:
            exist_y = True
            if y.ndim == 1:
                y = y.reshape(-1, 1)

        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError("X와 y는 numpy.ndarray여야 합니다.")
        if X.ndim != 2 or y.ndim != 2:
            raise ValueError("X와 y는 2차원 배열이어야 합니다.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X와 y의 샘플 수가 일치해야 합니다.")

        uniq = np.unique(y)
        if np.array_equal(uniq, [1]) or np.array_equal(uniq, [-1]):
            y_type = "single"
        elif set(uniq.tolist()) == {1, -1}:
            y_type = "hybrid"
        else:
            raise ValueError("y는 1(정상) 또는 -1(이상치) 값만 가질 수 있습니다.")

        return X, y, y_type, exist_y


def labels_to_pm1(y: np.ndarray) -> np.ndarray:
    """이상치 탐지 라벨(0=정상, 1=이상치)을 SVDD가 요구하는 형식(1=정상, -1=이상치)으로 변환."""
    return np.where(y == 1, -1, 1)


def predictions_to_binary(pred: np.ndarray) -> np.ndarray:
    """SVDD predict() 결과({1, -1})를 이상치 탐지 라벨(0=정상, 1=이상치)로 변환."""
    return np.where(pred == -1, 1, 0)
