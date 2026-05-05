from __future__ import annotations

from pathlib import Path
import sys
import importlib.util

import pytest


ROOT = Path(__file__).resolve().parents[1]

EVAL_SPEC = importlib.util.spec_from_file_location(
    "evaluate_rally_proposals_run",
    ROOT / "tasks/evaluate_rally_proposals/run.py",
)
eval_rally_run = importlib.util.module_from_spec(EVAL_SPEC)
assert EVAL_SPEC.loader is not None
EVAL_SPEC.loader.exec_module(eval_rally_run)


def test_slim_row_keeps_debug_fields() -> None:
    row = {
        "proposal_id": 7,
        "start_frame": 100,
        "end_frame": 140,
        "duration_frames": 41,
        "event_count": 1,
        "bounce_count": 1,
        "hit_count": 0,
        "serve_like_start": False,
        "serve_like_score": 0.22,
        "toss_like_start": True,
        "toss_rise_px": 18.0,
        "toss_x_span_px": 9.0,
        "rule_keep": False,
        "rule_reject_reason": "low_event_non_serve",
        "label": "true_rally",
    }

    slim = eval_rally_run.slim_row(row, label_field="label")

    assert slim["proposal_id"] == 7
    assert slim["toss_like_start"] is True
    assert slim["label"] == "true_rally"


def test_f1_handles_basic_counts() -> None:
    assert eval_rally_run.f1(8, 2, 2) == pytest.approx(0.8)


def test_build_positive_groups_splits_on_handoff() -> None:
    rows = [
        {"proposal_id": 1, "start_frame": 10, "end_frame": 20, "event_count": 1, "label": "true_rally"},
        {"proposal_id": 2, "start_frame": 21, "end_frame": 30, "event_count": 0, "label": "true_rally"},
        {"proposal_id": 3, "start_frame": 31, "end_frame": 40, "event_count": 2, "label": "handoff"},
        {"proposal_id": 4, "start_frame": 41, "end_frame": 50, "event_count": 3, "label": "serve_miss_or_ace"},
    ]

    groups = eval_rally_run.build_positive_groups(
        rows,
        label_field="label",
        positive_labels={"true_rally", "serve_miss_or_ace"},
    )

    assert [group["proposal_ids"] for group in groups] == [[1, 2], [4]]
    assert groups[0]["event_count"] == 1
