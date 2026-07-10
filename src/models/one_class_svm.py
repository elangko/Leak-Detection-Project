"""One-Class SVM(OC-SVM) 이상탐지 모델.

One_class_SVM.ipynb 정리본. 논문 2.1절 OC-SVM 모형에 해당한다.
scikit-learn의 OneClassSVM을 감싸서, 검증셋 F1 스코어 기준으로 (nu, gamma)를
grid search하는 기능을 추가했다.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score
from sklearn.svm import OneClassSVM


def anomaly_score(model: OneClassSVM, X: np.ndarray) -> np.ndarray:
    """점수가 높을수록 이상치에 가깝도록 부호를 뒤집는다.

    OneClassSVM.score_samples는 값이 클수록 "정상"에 가깝다는 뜻이므로,
    다른 모형들과 동일하게 '높을수록 이상치'가 되도록 부호를 반전한다.
    (원본 노트북의 1/score 변환은 score가 0에 가까울 때 불안정해서 사용하지 않는다.)
    """
    return -model.score_samples(X)


def predict_labels(model: OneClassSVM, X: np.ndarray) -> np.ndarray:
    """OneClassSVM.predict()의 {-1, 1}을 {이상치=1, 정상=0}으로 변환한다."""
    raw = model.predict(X)
    return np.where(raw == -1, 1, 0)


def grid_search(
    X_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    nu_list: list[float] | None = None,
    gamma_list: list[float] | None = None,
) -> tuple[OneClassSVM, dict]:
    """검증셋 F1 스코어를 기준으로 최적의 (nu, gamma) 조합을 찾는다."""
    nu_list = nu_list or [0.01, 0.05, 0.1, 0.2]
    gamma_list = gamma_list or [1e-4, 1e-5, 1e-6, 1e-7]

    best_f1 = -1.0
    best_model = None
    best_params = {}

    for nu in nu_list:
        for gamma in gamma_list:
            model = OneClassSVM(nu=nu, kernel="rbf", gamma=gamma)
            model.fit(X_train)
            pred = predict_labels(model, X_valid)
            f1 = f1_score(y_valid, pred, pos_label=1)
            if f1 > best_f1:
                best_f1 = f1
                best_model = model
                best_params = {"nu": nu, "gamma": gamma, "f1": f1}
            print(f"[OC-SVM] nu={nu}, gamma={gamma} -> valid F1={f1:.4f}")

    print(f"[OC-SVM] best params: {best_params}")
    return best_model, best_params
