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


@dataclass
class MLPClassifier:
    classes: list[str]
    input_weights: np.ndarray
    hidden_bias: np.ndarray
    output_weights: np.ndarray
    output_bias: np.ndarray
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
        hidden_units: int = 64,
        epochs: int = 900,
        learning_rate: float = 0.01,
        l2: float = 0.0005,
        seed: int = 42,
    ) -> "MLPClassifier":
        rng = np.random.default_rng(seed)
        scaler = StandardScaler.fit(x)
        x_scaled = scaler.transform(x)
        input_weights = rng.normal(0.0, np.sqrt(2.0 / x_scaled.shape[1]), size=(x_scaled.shape[1], hidden_units))
        hidden_bias = np.zeros(hidden_units, dtype=np.float64)
        output_weights = rng.normal(0.0, np.sqrt(2.0 / hidden_units), size=(hidden_units, len(classes)))
        output_bias = np.zeros(len(classes), dtype=np.float64)
        class_weights = _balanced_class_weights(y, len(classes))
        y_onehot = np.eye(len(classes), dtype=np.float64)[y]
        sample_weights = class_weights[y]

        for _ in range(epochs):
            hidden_linear = x_scaled @ input_weights + hidden_bias
            hidden = _relu(hidden_linear)
            probabilities = _softmax(hidden @ output_weights + output_bias)
            output_error = (probabilities - y_onehot) * sample_weights[:, None]
            output_weights_gradient = (hidden.T @ output_error) / x_scaled.shape[0] + l2 * output_weights
            output_bias_gradient = output_error.mean(axis=0)
            hidden_error = (output_error @ output_weights.T) * (hidden_linear > 0.0)
            input_weights_gradient = (x_scaled.T @ hidden_error) / x_scaled.shape[0] + l2 * input_weights
            hidden_bias_gradient = hidden_error.mean(axis=0)
            input_weights -= learning_rate * input_weights_gradient
            hidden_bias -= learning_rate * hidden_bias_gradient
            output_weights -= learning_rate * output_weights_gradient
            output_bias -= learning_rate * output_bias_gradient
        return cls(
            classes=classes,
            input_weights=input_weights,
            hidden_bias=hidden_bias,
            output_weights=output_weights,
            output_bias=output_bias,
            scaler=scaler,
            feature_names=feature_names,
        )

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        hidden = _relu(self.scaler.transform(x) @ self.input_weights + self.hidden_bias)
        return _softmax(hidden @ self.output_weights + self.output_bias)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(x), axis=1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": "mlp_classifier",
            "classes": self.classes,
            "feature_names": self.feature_names,
            "input_weights": self.input_weights.tolist(),
            "hidden_bias": self.hidden_bias.tolist(),
            "output_weights": self.output_weights.tolist(),
            "output_bias": self.output_bias.tolist(),
            "scaler": self.scaler.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MLPClassifier":
        return cls(
            classes=list(payload["classes"]),
            input_weights=np.asarray(payload["input_weights"], dtype=np.float64),
            hidden_bias=np.asarray(payload["hidden_bias"], dtype=np.float64),
            output_weights=np.asarray(payload["output_weights"], dtype=np.float64),
            output_bias=np.asarray(payload["output_bias"], dtype=np.float64),
            scaler=StandardScaler.from_dict(payload["scaler"]),
            feature_names=list(payload["feature_names"]),
        )


@dataclass
class DenseMLPClassifier:
    classes: list[str]
    layer_weights: list[np.ndarray]
    layer_biases: list[np.ndarray]
    scaler: StandardScaler
    feature_names: list[str]

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        values = self.scaler.transform(x)
        for index, (weights, bias) in enumerate(zip(self.layer_weights, self.layer_biases, strict=True)):
            values = values @ weights + bias
            if index < len(self.layer_weights) - 1:
                values = _relu(values)
        return _softmax(values)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(x), axis=1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": "dense_mlp_classifier",
            "classes": self.classes,
            "feature_names": self.feature_names,
            "layer_weights": [weights.tolist() for weights in self.layer_weights],
            "layer_biases": [bias.tolist() for bias in self.layer_biases],
            "scaler": self.scaler.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DenseMLPClassifier":
        return cls(
            classes=list(payload["classes"]),
            layer_weights=[np.asarray(weights, dtype=np.float64) for weights in payload["layer_weights"]],
            layer_biases=[np.asarray(bias, dtype=np.float64) for bias in payload["layer_biases"]],
            scaler=StandardScaler.from_dict(payload["scaler"]),
            feature_names=list(payload["feature_names"]),
        )


EventClassifier = SoftmaxRegression | MLPClassifier | DenseMLPClassifier


def load_event_classifier(payload: dict[str, Any]) -> EventClassifier:
    model_type = payload.get("model_type", "softmax_regression")
    if model_type == "softmax_regression":
        return SoftmaxRegression.from_dict(payload)
    if model_type == "mlp_classifier":
        return MLPClassifier.from_dict(payload)
    if model_type == "dense_mlp_classifier":
        return DenseMLPClassifier.from_dict(payload)
    raise ValueError(f"Unsupported event classifier model_type: {model_type}")


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _relu(values: np.ndarray) -> np.ndarray:
    return np.maximum(values, 0.0)


def _balanced_class_weights(y: np.ndarray, class_count: int) -> np.ndarray:
    counts = np.bincount(y, minlength=class_count).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = len(y) / (class_count * counts)
    return weights / weights.mean()
