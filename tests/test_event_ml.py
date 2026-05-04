from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datapingpong.events.ml import SoftmaxRegression


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
