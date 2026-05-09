from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datapingpong.events.ml import SoftmaxRegression

TRAIN_EVENT_SPEC = importlib.util.spec_from_file_location("train_event_classifier_run", ROOT / "tasks/train_event_classifier/run.py")
train_event_run = importlib.util.module_from_spec(TRAIN_EVENT_SPEC)
assert TRAIN_EVENT_SPEC.loader is not None
TRAIN_EVENT_SPEC.loader.exec_module(train_event_run)


def test_softmax_regression_learns_separable_classes() -> None:
    x = np.asarray(
        [
            [-2.0, -1.0],
            [-1.5, -1.0],
            [1.0, 1.0],
            [1.5, 1.0],
            [3.0, -1.0],
            [3.5, -1.0],
        ]
    )
    y = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)

    model = SoftmaxRegression.fit(x, y, classes=["a", "b", "c"], feature_names=["x0", "x1"], epochs=400)

    assert model.predict(x).tolist() == y.tolist()
    restored = SoftmaxRegression.from_dict(model.to_dict())
    assert restored.predict(x).tolist() == y.tolist()


def test_validate_prediction_events_rejects_untrained_class() -> None:
    train_event_run.validate_prediction_events(["bounce"], ["none", "bounce", "net_hit"])
    try:
        train_event_run.validate_prediction_events(["hit"], ["none", "bounce", "net_hit"])
    except ValueError as error:
        assert "Invalid" in str(error)
    else:
        raise AssertionError("Expected invalid prediction event to fail")
