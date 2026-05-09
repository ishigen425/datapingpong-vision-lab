from __future__ import annotations

from pathlib import Path
import importlib.util
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SPEC = importlib.util.spec_from_file_location(
    "train_bounce_classifier_run",
    ROOT / "tasks/train_bounce_classifier/run.py",
)
bounce_run = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(bounce_run)


def test_nms_keeps_highest_probability_per_window() -> None:
    candidates = [(10, 0.5), (12, 0.9), (25, 0.4)]

    kept = bounce_run.nms(candidates, window=4)

    assert kept == [(12, 0.9), (25, 0.4)]


def test_score_key_prefers_f1_then_precision_then_recall() -> None:
    assert bounce_run.score_key({"f1": 0.8, "precision": 0.7, "recall": 0.9}) > bounce_run.score_key(
        {"f1": 0.7, "precision": 0.99, "recall": 0.99}
    )
