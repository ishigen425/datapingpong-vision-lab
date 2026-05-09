from __future__ import annotations

from pathlib import Path
import importlib.util
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datapingpong.events import BallPoint


SPEC = importlib.util.spec_from_file_location(
    "annotate_bounce_video_run",
    ROOT / "tasks/annotate_bounce_video/run.py",
)
annotate_run = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(annotate_run)


def test_has_toss_like_context_detects_vertical_toss() -> None:
    points = {
        frame: BallPoint(frame, 100 + (frame % 2), y)
        for frame, y in enumerate([300, 260, 220, 180, 150, 170, 210, 250, 295])
    }

    assert annotate_run.has_toss_like_context(
        8,
        points,
        {"width": 200, "height": 400},
        ball_frame_width=200,
        ball_frame_height=400,
        lookback=8,
        max_x_span=20,
        min_y_rise=100,
        min_y_drop=100,
    )


def test_has_toss_like_context_rejects_wide_sideways_motion() -> None:
    points = {
        frame: BallPoint(frame, 30 + frame * 30, y)
        for frame, y in enumerate([300, 260, 220, 180, 150, 170, 210, 250, 295])
    }

    assert not annotate_run.has_toss_like_context(
        8,
        points,
        {"width": 400, "height": 400},
        ball_frame_width=400,
        ball_frame_height=400,
        lookback=8,
        max_x_span=20,
        min_y_rise=100,
        min_y_drop=100,
    )


def test_classify_candidates_marks_serve_bounce_before_weak() -> None:
    points = {
        frame: BallPoint(frame, 100 + (frame % 2), y)
        for frame, y in enumerate([300, 260, 220, 180, 150, 170, 210, 250, 295])
    }

    rows = annotate_run.classify_candidates(
        [(8, 0.32)],
        points,
        {"width": 200, "height": 400},
        ball_frame_width=200,
        ball_frame_height=400,
        strong_threshold=0.5,
        weak_threshold=0.3,
        serve_threshold=0.25,
        serve_lookback=8,
        serve_max_x_span=20,
        serve_min_y_rise=100,
        serve_min_y_drop=100,
    )

    assert rows == [(8, 0.32, "serve_bounce")]


def test_classify_candidates_drops_below_all_enabled_thresholds() -> None:
    points = {
        frame: BallPoint(frame, 100 + (frame % 2), y)
        for frame, y in enumerate([300, 260, 220, 180, 150, 170, 210, 250, 295])
    }

    rows = annotate_run.classify_candidates(
        [(8, 0.22)],
        points,
        {"width": 200, "height": 400},
        ball_frame_width=200,
        ball_frame_height=400,
        strong_threshold=0.5,
        weak_threshold=0.3,
        serve_threshold=0.25,
        serve_lookback=8,
        serve_max_x_span=20,
        serve_min_y_rise=100,
        serve_min_y_drop=100,
    )

    assert rows == []
