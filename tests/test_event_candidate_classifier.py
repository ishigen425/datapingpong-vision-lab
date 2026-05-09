from __future__ import annotations

from pathlib import Path
import importlib.util
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SPEC = importlib.util.spec_from_file_location(
    "train_event_candidate_classifier_run",
    ROOT / "tasks/train_event_candidate_classifier/run.py",
)
candidate_run = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(candidate_run)


def test_label_candidates_matches_predictions_once() -> None:
    candidates = [
        {"item": "__default__", "event": "hit", "frame": 100},
        {"item": "__default__", "event": "hit", "frame": 103},
        {"item": "__default__", "event": "hit", "frame": 160},
    ]
    reference = [{"item": "__default__", "event": "hit", "frame": 101}]

    labeled = candidate_run.label_candidates(candidates, reference, tolerance=4)

    assert [row["label"] for row in labeled] == ["hit", "none", "none"]


def test_build_candidate_row_adds_direction_features() -> None:
    candidate = {
        "item": "__default__",
        "event": "hit",
        "frame": 100,
        "probability": 0.8,
        "x": 100,
        "y": 200,
        "local_before_x_displacement": 40,
        "local_after_x_displacement": -20,
    }

    row = candidate_run.build_candidate_row(candidate, None, table_geometry=None)
    names = candidate_run.candidate_feature_names(include_table=False)
    values = dict(zip(names, row["features"], strict=True))

    assert values["candidate_is_hit"] == 1.0
    assert values["opposite_direction_x"] == 1.0
    assert values["same_direction_x"] == 0.0
    assert values["min_abs_directional_x"] == 20
