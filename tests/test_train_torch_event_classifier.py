from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SPEC = importlib.util.spec_from_file_location(
    "train_torch_event_classifier_run",
    ROOT / "tasks/train_torch_event_classifier/run.py",
)
torch_run = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(torch_run)


def test_balanced_class_weights_normalizes_mean_weight() -> None:
    weights = torch_run.balanced_class_weights(np.asarray([0, 0, 0, 1], dtype=np.int64), 2)

    assert weights[1] > weights[0]
    assert np.isclose(weights.mean(), 1.0)


def test_export_model_matches_torch_forward_without_dropout() -> None:
    network = torch_run.TorchMLP(input_size=2, hidden_units=[3], output_size=2, dropout=0.0)
    linear_layers = [layer for layer in network.network if isinstance(layer, nn.Linear)]
    with torch.no_grad():
        linear_layers[0].weight.copy_(torch.tensor([[0.2, -0.1], [0.5, 0.3], [-0.4, 0.7]]))
        linear_layers[0].bias.copy_(torch.tensor([0.1, -0.2, 0.3]))
        linear_layers[1].weight.copy_(torch.tensor([[0.6, -0.3, 0.2], [-0.2, 0.4, 0.5]]))
        linear_layers[1].bias.copy_(torch.tensor([0.05, -0.05]))

    x = np.asarray([[1.0, 2.0], [3.0, -1.0]], dtype=np.float64)
    scaler = torch_run.StandardScaler.fit(x)
    exported = torch_run.export_model(network, scaler=scaler, classes=["none", "bounce"], feature_names=["x0", "x1"])

    torch_logits = network(torch.as_tensor(scaler.transform(x), dtype=torch.float32)).detach().numpy()
    torch_probabilities = np.exp(torch_logits - torch_logits.max(axis=1, keepdims=True))
    torch_probabilities = torch_probabilities / torch_probabilities.sum(axis=1, keepdims=True)

    assert np.allclose(exported.predict_proba(x), torch_probabilities, atol=1e-6)


def test_sequence_for_index_pads_edges() -> None:
    features = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)

    sequence = torch_run.sequence_for_index(features, center_index=0, sequence_radius=1)

    assert sequence.tolist() == [[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]]
