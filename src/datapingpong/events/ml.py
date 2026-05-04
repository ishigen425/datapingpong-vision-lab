from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class StandardScaler:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "StandardScaler":
        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale = np.where(scale < 1e-8, 1.0, scale)
        return cls(mean=mean, scale=scale)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.scale

    def to_dict(self) -> dict[str, Any]:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StandardScaler":
        return cls(mean=np.asarray(payload["mean"], dtype=np.float64), scale=np.asarray(payload["scale"], dtype=np.float64))


@dataclass
class SoftmaxRegression:
    classes: list[str]
    weights: np.ndarray
    bias: np.ndarray
    scaler: StandardScaler
    feature_names: list[str]

    @classmethod
    def fit(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        *,
        classes: list[str],
        feature_names: list[str],
        epochs: int = 600,
        learning_rate: float = 0.08,
        l2: float = 0.001,
        seed: int = 42,
    ) -> "SoftmaxRegression":
        rng = np.random.default_rng(seed)
        scaler = StandardScaler.fit(x)
        x_scaled = scaler.transform(x)
        weights = rng.normal(0.0, 0.01, size=(x_scaled.shape[1], len(classes)))
        bias = np.zeros(len(classes), dtype=np.float64)
        class_weights = _balanced_class_weights(y, len(classes))
        y_onehot = np.eye(len(classes), dtype=np.float64)[y]
        sample_weights = class_weights[y]

        for _ in range(epochs):
            probabilities = _softmax(x_scaled @ weights + bias)
            error = (probabilities - y_onehot) * sample_weights[:, None]
            weights_gradient = (x_scaled.T @ error) / x_scaled.shape[0] + l2 * weights
            bias_gradient = error.mean(axis=0)
            weights -= learning_rate * weights_gradient
            bias -= learning_rate * bias_gradient
        return cls(classes=classes, weights=weights, bias=bias, scaler=scaler, feature_names=feature_names)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return _softmax(self.scaler.transform(x) @ self.weights + self.bias)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(x), axis=1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": "softmax_regression",
            "classes": self.classes,
            "feature_names": self.feature_names,
            "weights": self.weights.tolist(),
            "bias": self.bias.tolist(),
            "scaler": self.scaler.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SoftmaxRegression":
        return cls(
            classes=list(payload["classes"]),
            weights=np.asarray(payload["weights"], dtype=np.float64),
            bias=np.asarray(payload["bias"], dtype=np.float64),
            scaler=StandardScaler.from_dict(payload["scaler"]),
            feature_names=list(payload["feature_names"]),
        )


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _balanced_class_weights(y: np.ndarray, class_count: int) -> np.ndarray:
    counts = np.bincount(y, minlength=class_count).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = len(y) / (class_count * counts)
    return weights / weights.mean()
