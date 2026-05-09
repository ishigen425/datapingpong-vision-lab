from __future__ import annotations

from pathlib import Path
import sys
import importlib.util


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tasks/evaluate_event_detection"))

from datapingpong.events import BallPoint, detect_event_peaks, score_trajectory
from datapingpong.events.table import table_geometry_from_dict
from datapingpong.events.trajectory import EventPeak
from run import match_frames, summarize

DETECT_EVENTS_SPEC = importlib.util.spec_from_file_location("detect_events_run", ROOT / "tasks/detect_events_from_ball/run.py")
detect_events_run = importlib.util.module_from_spec(DETECT_EVENTS_SPEC)
assert DETECT_EVENTS_SPEC.loader is not None
DETECT_EVENTS_SPEC.loader.exec_module(detect_events_run)


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


def test_hit_probability_peaks_on_x_velocity_flip() -> None:
    points = [
        BallPoint(0, 0, 100),
        BallPoint(1, 10, 100),
        BallPoint(2, 20, 100),
        BallPoint(3, 10, 102),
        BallPoint(4, 0, 104),
    ]

    rows = score_trajectory(points, smooth_window=1)

    peak = max(rows, key=lambda row: row.hit_probability)
    assert peak.frame == 2
    assert peak.hit_probability > 0.5


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


def test_table_arbitration_keeps_inside_bounce_over_inside_hit() -> None:
    geometry = table_geometry_from_dict(
        {
            "corners": {
                "far_left": [0, 0],
                "far_right": [100, 0],
                "near_right": [100, 100],
                "near_left": [0, 100],
            }
        }
    )
    peaks = [
        EventPeak("bounce", 100, 0.7, 50, 50),
        EventPeak("hit", 103, 0.8, 52, 52),
    ]

    kept = detect_events_run.arbitrate_table_events(peaks, geometry, window=6)

    assert [(peak.event, peak.frame) for peak in kept] == [("bounce", 100)]


def test_table_arbitration_keeps_outside_hit_over_edge_bounce() -> None:
    geometry = table_geometry_from_dict(
        {
            "corners": {
                "far_left": [0, 0],
                "far_right": [100, 0],
                "near_right": [100, 100],
                "near_left": [0, 100],
            }
        }
    )
    peaks = [
        EventPeak("bounce", 100, 0.8, 102, 50),
        EventPeak("hit", 103, 0.7, 115, 52),
    ]

    kept = detect_events_run.arbitrate_table_events(peaks, geometry, window=6)

    assert [(peak.event, peak.frame) for peak in kept] == [("hit", 103)]


def test_hit_local_motion_filter_removes_stationary_flip_candidate() -> None:
    points = [
        BallPoint(94, 100, 100),
        BallPoint(95, 101, 100),
        BallPoint(96, 99, 101),
        BallPoint(97, 100, 100),
        BallPoint(98, 101, 100),
        BallPoint(99, 99, 101),
        BallPoint(100, 100, 100),
        BallPoint(101, 101, 100),
        BallPoint(102, 99, 101),
        BallPoint(103, 100, 100),
        BallPoint(104, 101, 100),
        BallPoint(105, 99, 101),
        BallPoint(106, 100, 100),
    ]
    peaks = [
        EventPeak("hit", 100, 0.8, 100, 100),
        EventPeak("bounce", 101, 0.7, 101, 100),
    ]

    kept = detect_events_run.filter_hits_by_local_motion(
        peaks,
        points,
        window=6,
        min_x_span=20,
        min_y_span=0,
        min_detections=4,
        min_directional_x_displacement=0,
    )

    assert [(peak.event, peak.frame) for peak in kept] == [("bounce", 101)]


def test_hit_local_motion_filter_keeps_wide_x_motion_candidate() -> None:
    points = [
        BallPoint(94, 40, 100),
        BallPoint(95, 55, 100),
        BallPoint(96, 70, 101),
        BallPoint(97, 85, 101),
        BallPoint(98, 100, 102),
        BallPoint(99, 115, 102),
        BallPoint(100, 130, 103),
        BallPoint(101, 115, 103),
        BallPoint(102, 100, 104),
        BallPoint(103, 85, 104),
        BallPoint(104, 70, 105),
        BallPoint(105, 55, 105),
        BallPoint(106, 40, 106),
    ]
    peaks = [EventPeak("hit", 100, 0.8, 130, 103)]

    kept = detect_events_run.filter_hits_by_local_motion(
        peaks,
        points,
        window=6,
        min_x_span=20,
        min_y_span=0,
        min_detections=4,
        min_directional_x_displacement=0,
    )

    assert [(peak.event, peak.frame) for peak in kept] == [("hit", 100)]
    assert kept[0].local_x_span == 90
    assert kept[0].local_detections == 13


def test_hit_local_motion_filter_removes_single_frame_x_jitter() -> None:
    points = [
        BallPoint(94, 50, 100),
        BallPoint(95, 60, 100),
        BallPoint(96, 70, 100),
        BallPoint(97, 80, 100),
        BallPoint(98, 90, 100),
        BallPoint(99, 100, 100),
        BallPoint(100, 110, 100),
        BallPoint(101, 100, 101),
        BallPoint(102, 120, 101),
        BallPoint(103, 130, 101),
        BallPoint(104, 140, 101),
        BallPoint(105, 150, 101),
        BallPoint(106, 160, 101),
    ]
    peaks = [EventPeak("hit", 100, 0.8, 110, 100)]

    kept = detect_events_run.filter_hits_by_local_motion(
        peaks,
        points,
        window=6,
        min_x_span=20,
        min_y_span=0,
        min_detections=4,
        min_directional_x_displacement=8,
    )

    assert kept == []


def test_hit_local_motion_filter_rejects_strong_same_direction_pass() -> None:
    points = [
        BallPoint(94, 40, 100),
        BallPoint(95, 55, 100),
        BallPoint(96, 70, 101),
        BallPoint(97, 85, 101),
        BallPoint(98, 100, 102),
        BallPoint(99, 115, 102),
        BallPoint(100, 130, 103),
        BallPoint(101, 145, 103),
        BallPoint(102, 160, 104),
        BallPoint(103, 175, 104),
        BallPoint(104, 190, 105),
        BallPoint(105, 205, 105),
        BallPoint(106, 220, 106),
    ]
    peaks = [EventPeak("hit", 100, 0.8, 130, 103)]

    kept = detect_events_run.filter_hits_by_local_motion(
        peaks,
        points,
        window=6,
        min_x_span=20,
        min_y_span=0,
        min_detections=4,
        min_directional_x_displacement=0,
        reject_same_directional_x_displacement=30,
    )

    assert kept == []


def test_hit_local_motion_filter_keeps_weak_same_direction_candidate() -> None:
    points = [
        BallPoint(94, 100, 100),
        BallPoint(95, 108, 100),
        BallPoint(96, 116, 101),
        BallPoint(97, 124, 101),
        BallPoint(98, 132, 102),
        BallPoint(99, 140, 102),
        BallPoint(100, 148, 103),
        BallPoint(101, 150, 103),
        BallPoint(102, 152, 104),
        BallPoint(103, 154, 104),
        BallPoint(104, 156, 105),
        BallPoint(105, 158, 105),
        BallPoint(106, 160, 106),
    ]
    peaks = [EventPeak("hit", 100, 0.8, 148, 103)]

    kept = detect_events_run.filter_hits_by_local_motion(
        peaks,
        points,
        window=6,
        min_x_span=20,
        min_y_span=0,
        min_detections=4,
        min_directional_x_displacement=0,
        reject_same_directional_x_displacement=30,
    )

    assert [(peak.event, peak.frame) for peak in kept] == [("hit", 100)]


def test_hit_local_motion_filter_keeps_sustained_x_reversal() -> None:
    points = [
        BallPoint(94, 40, 100),
        BallPoint(95, 55, 100),
        BallPoint(96, 70, 101),
        BallPoint(97, 85, 101),
        BallPoint(98, 100, 102),
        BallPoint(99, 115, 102),
        BallPoint(100, 130, 103),
        BallPoint(101, 115, 103),
        BallPoint(102, 100, 104),
        BallPoint(103, 85, 104),
        BallPoint(104, 70, 105),
        BallPoint(105, 55, 105),
        BallPoint(106, 40, 106),
    ]
    peaks = [EventPeak("hit", 100, 0.8, 130, 103)]

    kept = detect_events_run.filter_hits_by_local_motion(
        peaks,
        points,
        window=6,
        min_x_span=20,
        min_y_span=0,
        min_detections=4,
        min_directional_x_displacement=8,
    )

    assert [(peak.event, peak.frame) for peak in kept] == [("hit", 100)]
    assert kept[0].local_before_x_displacement == 90
    assert kept[0].local_after_x_displacement == -90


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
