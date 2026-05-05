from __future__ import annotations

from pathlib import Path
import sys
import importlib.util

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datapingpong.events.rally import RallySegment
from datapingpong.events.rally_ml import build_rally_proposal_feature_table, labeled_rows_to_matrix


TRAIN_SPEC = importlib.util.spec_from_file_location(
    "train_rally_start_classifier_run",
    ROOT / "tasks/train_rally_start_classifier/run.py",
)
train_rally_run = importlib.util.module_from_spec(TRAIN_SPEC)
assert TRAIN_SPEC.loader is not None
TRAIN_SPEC.loader.exec_module(train_rally_run)


def test_build_rally_proposal_feature_table_exports_rule_and_toss_features() -> None:
    proposals = [
        RallySegment(
            start_frame=10,
            end_frame=30,
            track_start_frame=12,
            track_end_frame=28,
            detected_points=14,
            min_x=100.0,
            max_x=130.0,
            min_y=80.0,
            max_y=120.0,
            events=({"event": "bounce", "frame": 20},),
            serve_like_start=True,
            serve_like_score=0.55,
            pose_evidence_frames=20,
            toss_like_start=True,
            toss_rise_px=18.0,
            toss_x_span_px=9.0,
            rule_keep=True,
            rule_reject_reason=None,
        )
    ]

    table = build_rally_proposal_feature_table(proposals)

    assert table.feature_names[0] == "duration_frames"
    assert table.rows[0]["rule_keep"] is True
    assert table.rows[0]["toss_like_start"] is True
    assert table.rows[0]["first_event_offset"] == 8
    assert table.features.shape == (1, len(table.feature_names))


def test_labeled_rows_to_matrix_filters_unlabeled_rows() -> None:
    rows = [
        {"duration_frames": 100, "event_count": 2, "label": "true_rally"},
        {"duration_frames": 40, "event_count": 0, "label": "handoff"},
        {"duration_frames": 55, "event_count": 1, "label": None},
    ]

    x, y, classes = labeled_rows_to_matrix(rows, ["duration_frames", "event_count"])

    assert x.shape == (2, 2)
    assert set(classes) == {"handoff", "true_rally"}
    assert y.dtype == np.int64


def test_split_indices_keeps_non_empty_train_and_eval() -> None:
    train_indices, eval_indices = train_rally_run.split_indices(6, train_fraction=0.8, seed=42)

    assert len(train_indices) == 5
    assert len(eval_indices) == 1
