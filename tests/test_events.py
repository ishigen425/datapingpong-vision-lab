from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tasks/evaluate_event_detection"))

from datapingpong.events import BallPoint, detect_event_peaks, score_trajectory
from run import match_frames, summarize


def test_bounce_probability_peaks_on_y_velocity_flip() -> None:
    points = [
        BallPoint(0, 0, 80),
        BallPoint(1, 5, 90),
        BallPoint(2, 10, 105),
        BallPoint(3, 15, 90),
        BallPoint(4, 20, 80),
    ]

    rows = score_trajectory(points, smooth_window=1)

    peak = max(rows, key=lambda row: row.bounce_probability)
    assert peak.frame == 2
    assert peak.bounce_probability > 0.55


def test_hit_probability_peaks_on_sharp_direction_change() -> None:
    points = [
        BallPoint(0, 0, 100),
        BallPoint(1, 10, 100),
        BallPoint(2, 20, 100),
        BallPoint(3, 20, 110),
        BallPoint(4, 20, 120),
    ]

    rows = score_trajectory(points, smooth_window=1)

    peak = max(rows, key=lambda row: row.hit_probability)
    assert peak.frame in {1, 2, 3}
    assert peak.hit_probability > 0.4


def test_detect_event_peaks_applies_per_event_nms() -> None:
    points = [
        BallPoint(0, 0, 80),
        BallPoint(1, 5, 90),
        BallPoint(2, 10, 105),
        BallPoint(3, 15, 90),
        BallPoint(4, 20, 80),
        BallPoint(12, 20, 80),
        BallPoint(13, 25, 90),
        BallPoint(14, 30, 105),
        BallPoint(15, 35, 90),
        BallPoint(16, 40, 80),
    ]
    rows = score_trajectory(points, smooth_window=1)

    peaks = detect_event_peaks(rows, bounce_threshold=0.5, hit_threshold=1.1, nms_window=4)

    assert [peak.event for peak in peaks] == ["bounce", "bounce"]
    assert [peak.frame for peak in peaks] == [2, 14]


def test_match_frames_uses_nearest_unmatched_reference() -> None:
    matches, false_positive, false_negative = match_frames([10, 20], [7, 21, 40], tolerance=4)

    assert matches == [(10, 7), (20, 21)]
    assert false_positive == [40]
    assert false_negative == []


def test_summarize_reports_micro_metrics() -> None:
    summary = summarize(
        reference=[{"event": "bounce", "frame": 10}, {"event": "hit", "frame": 30}],
        predictions=[{"event": "bounce", "frame": 12}, {"event": "hit", "frame": 50}],
        tolerance=4,
        event_names=["bounce", "hit"],
    )

    assert summary["events"]["bounce"]["true_positive"] == 1
    assert summary["events"]["hit"]["false_positive"] == 1
    assert summary["micro"]["precision"] == 0.5
