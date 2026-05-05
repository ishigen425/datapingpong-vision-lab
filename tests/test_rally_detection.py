from __future__ import annotations

from pathlib import Path
import json
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datapingpong.events import BallPoint, build_rally_proposals, detect_rallies
from datapingpong.events.io import load_reference_events


def test_detect_rallies_splits_long_ball_gap() -> None:
    points = [
        BallPoint(10, 100, 80),
        BallPoint(11, 102, 82),
        BallPoint(12, 104, 83),
        BallPoint(13, 106, 84),
        BallPoint(14, 108, 85),
        BallPoint(15, 110, 86),
        BallPoint(16, 112, 87),
        BallPoint(17, 114, 88),
        BallPoint(18, 116, 89),
        BallPoint(19, 118, 90),
        BallPoint(20, 120, 91),
        BallPoint(21, 122, 92),
        BallPoint(40, 200, 110),
        BallPoint(41, 202, 112),
        BallPoint(42, 204, 114),
        BallPoint(43, 206, 116),
        BallPoint(44, 208, 118),
        BallPoint(45, 210, 120),
        BallPoint(46, 212, 122),
        BallPoint(47, 214, 124),
        BallPoint(48, 216, 126),
        BallPoint(49, 218, 128),
        BallPoint(50, 220, 130),
        BallPoint(51, 222, 132),
    ]
    events = [
        {"event": "bounce", "frame": 18, "x": 116, "y": 89},
        {"event": "hit", "frame": 48, "x": 216, "y": 126},
    ]

    rallies = detect_rallies(
        points,
        events,
        max_ball_gap=4,
        min_detected_points=6,
        min_span_frames=6,
        pre_padding=2,
        post_padding=3,
        merge_gap=3,
    )

    assert [(rally.start_frame, rally.end_frame) for rally in rallies] == [(8, 24), (38, 54)]
    assert [len(rally.events) for rally in rallies] == [1, 1]


def test_detect_rallies_merges_brief_interruption() -> None:
    points = [
        BallPoint(frame, 100 + frame, 80 + frame)
        for frame in list(range(100, 110)) + list(range(116, 126))
    ]

    rallies = detect_rallies(
        points,
        [],
        max_ball_gap=4,
        min_detected_points=6,
        min_span_frames=6,
        pre_padding=2,
        post_padding=2,
        merge_gap=10,
    )

    assert len(rallies) == 1
    assert rallies[0].track_start_frame == 100
    assert rallies[0].track_end_frame == 125
    assert rallies[0].detected_points == 20


def test_detect_rallies_merges_continued_kept_segments() -> None:
    points = [BallPoint(frame, 100 + frame, 80) for frame in range(100, 121)] + [
        BallPoint(frame, 240 + frame, 92) for frame in range(190, 211)
    ]
    events = [
        {"event": "bounce", "frame": 110, "x": 210, "y": 80},
        {"event": "hit", "frame": 200, "x": 440, "y": 92},
    ]

    rallies = detect_rallies(
        points,
        events,
        min_detected_points=6,
        min_span_frames=6,
        pre_padding=2,
        post_padding=2,
        merge_gap=10,
        continuation_gap=80,
    )

    assert len(rallies) == 1
    assert rallies[0].start_frame == 98
    assert rallies[0].end_frame == 212
    assert [event["event"] for event in rallies[0].events] == ["bounce", "hit"]


def test_build_rally_proposals_rejects_bounce_only_handoff() -> None:
    points = [BallPoint(frame, 800 + (frame - 100), 100) for frame in range(100, 241)]
    events = [
        {"event": "bounce", "frame": 130, "x": 800, "y": 100},
        {"event": "bounce", "frame": 160, "x": 950, "y": 100},
        {"event": "bounce", "frame": 190, "x": 1100, "y": 100},
    ]
    pose_features = {
        99: {
            "players": {
                "left": {"pose_detected": 1.0, "upper_body_visible_ratio": 1.0},
                "right": {"pose_detected": 1.0, "upper_body_visible_ratio": 1.0},
            }
        }
    }

    proposals = build_rally_proposals(points, events, pose_features, pre_padding=0, post_padding=0)

    assert len(proposals) == 1
    assert proposals[0].rule_keep is False
    assert proposals[0].rule_reject_reason == "bounce_only_handoff"


def test_build_rally_proposals_rejects_empty_serve_like_handoff() -> None:
    points = [
        BallPoint(50, 100, 120),
        BallPoint(51, 102, 112),
        BallPoint(52, 104, 105),
        BallPoint(53, 105, 99),
        BallPoint(54, 106, 96),
        BallPoint(55, 107, 98),
        BallPoint(56, 108, 102),
        BallPoint(57, 109, 108),
        BallPoint(58, 110, 114),
    ]
    pose_features = {
        frame: {
            "players": {
                "left": {
                    "pose_detected": 1.0,
                    "upper_body_visible_ratio": 1.0,
                    "shoulder_center_x": 0.3,
                    "shoulder_center_y": 0.4,
                    "hip_center_x": 0.3,
                    "hip_center_y": 0.5,
                    "shoulder_width": 0.08,
                    "hip_width": 0.08,
                    "torso_length": 0.1,
                    "left_wrist_rel_x": 0.0,
                    "left_wrist_rel_y": 0.0,
                    "right_wrist_rel_x": 0.0,
                    "right_wrist_rel_y": 0.0,
                    "left_wrist_velocity_x": 0.005,
                    "left_wrist_velocity_y": 0.0,
                    "right_wrist_velocity_x": 0.004,
                    "right_wrist_velocity_y": 0.0,
                },
                "right": {
                    "pose_detected": 1.0,
                    "upper_body_visible_ratio": 1.0,
                    "shoulder_center_x": 0.7,
                    "shoulder_center_y": 0.4,
                    "hip_center_x": 0.7,
                    "hip_center_y": 0.5,
                    "shoulder_width": 0.08,
                    "hip_width": 0.08,
                    "torso_length": 0.1,
                    "left_wrist_rel_x": 0.0,
                    "left_wrist_rel_y": 0.0,
                    "right_wrist_rel_x": 0.0,
                    "right_wrist_rel_y": 0.0,
                    "left_wrist_velocity_x": 0.003,
                    "left_wrist_velocity_y": 0.0,
                    "right_wrist_velocity_x": 0.004,
                    "right_wrist_velocity_y": 0.0,
                },
            }
        }
        for frame in range(14, 50)
    }

    proposals = build_rally_proposals(
        points,
        [],
        pose_features,
        min_detected_points=6,
        min_span_frames=6,
        no_event_max_detected_points=40,
        no_event_max_span_frames=60,
        serve_lookback_frames=24,
        serve_min_still_frames=8,
    )

    assert len(proposals) == 1
    assert proposals[0].serve_like_start is True
    assert proposals[0].rule_keep is False
    assert proposals[0].rule_reject_reason == "empty_serve_like_handoff"


def test_build_rally_proposals_rejects_long_no_event_handoff() -> None:
    points = [BallPoint(frame, 100 + min(frame - 100, 120), 100) for frame in range(100, 241)]
    pose_features = {
        99: {
            "players": {
                "left": {"pose_detected": 1.0, "upper_body_visible_ratio": 1.0},
                "right": {"pose_detected": 1.0, "upper_body_visible_ratio": 1.0},
            }
        }
    }

    proposals = build_rally_proposals(
        points,
        [],
        pose_features,
        pre_padding=0,
        post_padding=0,
        low_event_max_detected_points=20,
        low_event_max_span_frames=40,
    )

    assert len(proposals) == 1
    assert proposals[0].rule_keep is False
    assert proposals[0].rule_reject_reason == "long_no_event_handoff"


def test_build_rally_proposals_rejects_hit_only_handoff() -> None:
    points = [BallPoint(frame, 100 + min(frame - 100, 140), 100) for frame in range(100, 220)]
    events = [
        {"event": "hit", "frame": 130, "x": 180, "y": 100},
        {"event": "hit", "frame": 180, "x": 240, "y": 100},
    ]
    pose_features = {
        99: {
            "players": {
                "left": {"pose_detected": 1.0, "upper_body_visible_ratio": 1.0},
                "right": {"pose_detected": 1.0, "upper_body_visible_ratio": 1.0},
            }
        }
    }

    proposals = build_rally_proposals(points, events, pose_features, pre_padding=0, post_padding=0)

    assert len(proposals) == 1
    assert proposals[0].rule_keep is False
    assert proposals[0].rule_reject_reason == "hit_only_handoff"


def test_detect_rallies_filters_short_noise_segment() -> None:
    points = [
        BallPoint(10, 100, 100),
        BallPoint(11, 101, 100),
        BallPoint(12, 102, 100),
        BallPoint(30, 200, 120),
        BallPoint(31, 202, 122),
        BallPoint(32, 204, 124),
        BallPoint(33, 206, 126),
        BallPoint(34, 208, 128),
        BallPoint(35, 210, 130),
    ]

    rallies = detect_rallies(
        points,
        [],
        max_ball_gap=3,
        min_detected_points=4,
        min_span_frames=4,
        pre_padding=1,
        post_padding=1,
        merge_gap=2,
    )

    assert [(rally.track_start_frame, rally.track_end_frame) for rally in rallies] == [(30, 35)]


def test_load_reference_events_accepts_detect_events_payload(tmp_path: Path) -> None:
    path = tmp_path / "events.json"
    path.write_text(
        json.dumps(
            {
                "input": "dummy.json",
                "predicted_events": [
                    {"event": "bounce", "frame": 12, "x": 100, "y": 120},
                    {"event": "hit", "frame": 18, "x": 150, "y": 100},
                ],
            }
        ),
        encoding="utf-8",
    )

    events = load_reference_events(path)

    assert [(event["event"], event["frame"]) for event in events] == [("bounce", 12), ("hit", 18)]


def test_detect_rallies_drops_short_empty_segment_without_serve_like_start() -> None:
    points = [BallPoint(frame, 100 + frame, 100) for frame in range(50, 76)]
    pose_features = {
        frame: {
            "players": {
                "left": {
                    "pose_detected": 1.0,
                    "upper_body_visible_ratio": 1.0,
                    "shoulder_center_x": 0.3 + 0.02 * (frame - 40),
                    "shoulder_center_y": 0.4,
                    "hip_center_x": 0.3 + 0.02 * (frame - 40),
                    "hip_center_y": 0.5,
                    "left_wrist_velocity_x": 0.08,
                    "left_wrist_velocity_y": 0.0,
                    "right_wrist_velocity_x": 0.05,
                    "right_wrist_velocity_y": 0.0,
                },
                "right": {
                    "pose_detected": 1.0,
                    "upper_body_visible_ratio": 1.0,
                    "shoulder_center_x": 0.7 - 0.02 * (frame - 40),
                    "shoulder_center_y": 0.4,
                    "hip_center_x": 0.7 - 0.02 * (frame - 40),
                    "hip_center_y": 0.5,
                    "left_wrist_velocity_x": 0.06,
                    "left_wrist_velocity_y": 0.0,
                    "right_wrist_velocity_x": 0.07,
                    "right_wrist_velocity_y": 0.0,
                },
            }
        }
        for frame in range(14, 50)
    }

    rallies = detect_rallies(
        points,
        [],
        pose_features,
        min_detected_points=6,
        min_span_frames=6,
        no_event_max_detected_points=40,
        no_event_max_span_frames=60,
        serve_lookback_frames=24,
        serve_min_still_frames=8,
    )

    assert rallies == []


def test_detect_rallies_keeps_short_segment_with_serve_like_start_and_event() -> None:
    points = [
        BallPoint(50, 100, 120),
        BallPoint(51, 102, 112),
        BallPoint(52, 104, 105),
        BallPoint(53, 105, 99),
        BallPoint(54, 106, 96),
        BallPoint(55, 107, 98),
        BallPoint(56, 108, 102),
        BallPoint(57, 109, 108),
        BallPoint(58, 110, 114),
    ]
    pose_features = {
        frame: {
            "players": {
                "left": {
                    "pose_detected": 1.0,
                    "upper_body_visible_ratio": 1.0,
                    "shoulder_center_x": 0.3,
                    "shoulder_center_y": 0.4,
                    "hip_center_x": 0.3,
                    "hip_center_y": 0.5,
                    "left_wrist_velocity_x": 0.005,
                    "left_wrist_velocity_y": 0.0,
                    "right_wrist_velocity_x": 0.004,
                    "right_wrist_velocity_y": 0.0,
                },
                "right": {
                    "pose_detected": 1.0,
                    "upper_body_visible_ratio": 1.0,
                    "shoulder_center_x": 0.7,
                    "shoulder_center_y": 0.4,
                    "hip_center_x": 0.7,
                    "hip_center_y": 0.5,
                    "left_wrist_velocity_x": 0.003,
                    "left_wrist_velocity_y": 0.0,
                    "right_wrist_velocity_x": 0.004,
                    "right_wrist_velocity_y": 0.0,
                },
            }
        }
        for frame in range(14, 50)
    }

    rallies = detect_rallies(
        points,
        [{"event": "bounce", "frame": 55, "x": 107, "y": 98}],
        pose_features,
        min_detected_points=6,
        min_span_frames=6,
        no_event_max_detected_points=40,
        no_event_max_span_frames=60,
        serve_lookback_frames=24,
        serve_min_still_frames=8,
    )

    assert len(rallies) == 1
    assert rallies[0].serve_like_start is True
    assert rallies[0].toss_like_start is True


def test_detect_rallies_requires_serve_score_threshold() -> None:
    points = [
        BallPoint(50, 100, 120),
        BallPoint(51, 102, 112),
        BallPoint(52, 104, 105),
        BallPoint(53, 105, 99),
        BallPoint(54, 106, 96),
        BallPoint(55, 107, 98),
    ]
    pose_features = {
        frame: {
            "players": {
                "left": {
                    "pose_detected": 1.0,
                    "upper_body_visible_ratio": 1.0,
                    "shoulder_center_x": 0.3 + 0.001 * ((frame % 2) - 0.5),
                    "shoulder_center_y": 0.4,
                    "hip_center_x": 0.3 + 0.001 * ((frame % 2) - 0.5),
                    "hip_center_y": 0.5,
                    "left_wrist_velocity_x": 0.018,
                    "left_wrist_velocity_y": 0.0,
                    "right_wrist_velocity_x": 0.018,
                    "right_wrist_velocity_y": 0.0,
                },
                "right": {
                    "pose_detected": 1.0,
                    "upper_body_visible_ratio": 1.0,
                    "shoulder_center_x": 0.7 + 0.001 * ((frame % 2) - 0.5),
                    "shoulder_center_y": 0.4,
                    "hip_center_x": 0.7 + 0.001 * ((frame % 2) - 0.5),
                    "hip_center_y": 0.5,
                    "left_wrist_velocity_x": 0.018,
                    "left_wrist_velocity_y": 0.0,
                    "right_wrist_velocity_x": 0.018,
                    "right_wrist_velocity_y": 0.0,
                },
            }
        }
        for frame in range(14, 50)
    }

    rallies = detect_rallies(
        points,
        [],
        pose_features,
        min_detected_points=6,
        min_span_frames=6,
        serve_lookback_frames=24,
        serve_min_still_frames=8,
        serve_min_score=0.5,
    )

    assert rallies == []


def test_detect_rallies_requires_toss_like_start() -> None:
    points = [BallPoint(frame, 100 + frame, 120) for frame in range(50, 59)]
    pose_features = {
        frame: {
            "players": {
                "left": {
                    "pose_detected": 1.0,
                    "upper_body_visible_ratio": 1.0,
                    "shoulder_center_x": 0.3,
                    "shoulder_center_y": 0.4,
                    "hip_center_x": 0.3,
                    "hip_center_y": 0.5,
                    "left_wrist_velocity_x": 0.005,
                    "left_wrist_velocity_y": 0.0,
                    "right_wrist_velocity_x": 0.004,
                    "right_wrist_velocity_y": 0.0,
                },
                "right": {
                    "pose_detected": 1.0,
                    "upper_body_visible_ratio": 1.0,
                    "shoulder_center_x": 0.7,
                    "shoulder_center_y": 0.4,
                    "hip_center_x": 0.7,
                    "hip_center_y": 0.5,
                    "left_wrist_velocity_x": 0.003,
                    "left_wrist_velocity_y": 0.0,
                    "right_wrist_velocity_x": 0.004,
                    "right_wrist_velocity_y": 0.0,
                },
            }
        }
        for frame in range(14, 50)
    }

    rallies = detect_rallies(
        points,
        [],
        pose_features,
        min_detected_points=6,
        min_span_frames=6,
        serve_lookback_frames=24,
        serve_min_still_frames=8,
        serve_min_score=0.4,
        serve_min_toss_rise_px=12.0,
        serve_max_toss_x_span_px=24.0,
    )

    assert rallies == []


def test_detect_rallies_drops_short_low_event_non_serve_segment() -> None:
    points = [BallPoint(frame, 100 + frame, 100) for frame in range(50, 140)]
    events = [{"event": "bounce", "frame": 88}]
    pose_features = {
        frame: {
            "players": {
                "left": {
                    "pose_detected": 1.0,
                    "upper_body_visible_ratio": 1.0,
                    "shoulder_center_x": 0.2 + 0.01 * (frame - 20),
                    "shoulder_center_y": 0.4,
                    "hip_center_x": 0.2 + 0.01 * (frame - 20),
                    "hip_center_y": 0.5,
                    "left_wrist_velocity_x": 0.08,
                    "left_wrist_velocity_y": 0.0,
                    "right_wrist_velocity_x": 0.08,
                    "right_wrist_velocity_y": 0.0,
                },
                "right": {
                    "pose_detected": 1.0,
                    "upper_body_visible_ratio": 1.0,
                    "shoulder_center_x": 0.8 - 0.01 * (frame - 20),
                    "shoulder_center_y": 0.4,
                    "hip_center_x": 0.8 - 0.01 * (frame - 20),
                    "hip_center_y": 0.5,
                    "left_wrist_velocity_x": 0.08,
                    "left_wrist_velocity_y": 0.0,
                    "right_wrist_velocity_x": 0.08,
                    "right_wrist_velocity_y": 0.0,
                },
            }
        }
        for frame in range(14, 50)
    }

    rallies = detect_rallies(
        points,
        events,
        pose_features,
        min_detected_points=6,
        min_span_frames=6,
        low_event_max_events=1,
        low_event_max_detected_points=180,
        low_event_max_span_frames=220,
        serve_lookback_frames=24,
        serve_min_still_frames=8,
    )

    assert rallies == []
