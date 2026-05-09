from __future__ import annotations

from pathlib import Path
import sys
import importlib.util


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


ANNOTATE_SPEC = importlib.util.spec_from_file_location(
    "annotate_multimodal_run",
    ROOT / "tasks/annotate_multimodal_video/run.py",
)
annotate_multimodal_run = importlib.util.module_from_spec(ANNOTATE_SPEC)
assert ANNOTATE_SPEC.loader is not None
ANNOTATE_SPEC.loader.exec_module(annotate_multimodal_run)


def test_active_event_labels_respects_display_window() -> None:
    labels = annotate_multimodal_run.active_event_labels(
        [
            {"event": "bounce", "frame": 100, "probability": 0.7},
            {"event": "hit", "frame": 108, "probability": 0.8},
        ],
        103,
        window=4,
    )

    assert labels == ["BOUNCE f=100 p=0.70"]


def test_feature_text_lines_include_key_pose_values() -> None:
    lines = annotate_multimodal_run.feature_text_lines(
        {
            "players": {
                "left": {
                    "pose_detected": 1.0,
                    "upper_body_visible_ratio": 0.75,
                    "torso_tilt_degrees": -12.34,
                    "left_elbow_angle_degrees": 101.2,
                    "right_elbow_angle_degrees": 88.9,
                    "left_wrist_velocity_x": 0.123,
                    "left_wrist_velocity_y": -0.456,
                    "right_wrist_velocity_x": 0.3,
                    "right_wrist_velocity_y": 0.4,
                },
                "right": {
                    "pose_detected": 1.0,
                    "upper_body_visible_ratio": 0.55,
                    "torso_tilt_degrees": 6.78,
                    "left_elbow_angle_degrees": 120.0,
                    "right_elbow_angle_degrees": 95.0,
                    "left_wrist_velocity_x": -0.2,
                    "left_wrist_velocity_y": 0.1,
                    "right_wrist_velocity_x": 0.0,
                    "right_wrist_velocity_y": 0.2,
                },
            }
        }
    )

    assert lines[0] == "[left] detected=1 vis=0.75"
    assert "tilt=-12.3" in lines[1]
    assert "R_vel=(0.30, 0.40)" in lines[2]
    assert lines[3] == "[right] detected=1 vis=0.55"


def test_active_rally_returns_current_segment() -> None:
    rally = annotate_multimodal_run.active_rally(
        [
            {"id": 1, "start_frame": 10, "end_frame": 20, "duration_frames": 11, "event_counts": {"bounce": 1}},
            {"id": 2, "start_frame": 30, "end_frame": 50, "duration_frames": 21, "event_counts": {"hit": 2}},
        ],
        35,
    )

    assert rally is not None
    assert rally["id"] == 2


def test_load_ball_rows_preserves_unet_detection_fields(tmp_path: Path) -> None:
    path = tmp_path / "ball.json"
    path.write_text(
        '{"input_size":{"width":640,"height":360},"predictions":[{"frame":7,"x":10,"y":20,"confidence":0.4,"unet_x":30,"unet_y":40,"unet_confidence":0.9,"unet_detected":true}]}',
        encoding="utf-8",
    )

    rows = annotate_multimodal_run.load_ball_rows(path, {"width": 1920, "height": 1080}, frame_offset=4)

    assert rows[11]["x"] == 10.0
    assert rows[11]["unet_x"] == 30.0
    assert rows[11]["unet_y"] == 40.0
    assert rows[11]["unet_confidence"] == 0.9
    assert rows[11]["unet_detected"] is True
    assert annotate_multimodal_run.unet_point_from_ball_row(rows[11], {"width": 1920, "height": 1080}) == (90, 120)


def test_load_rallies_reads_payload_wrapper(tmp_path: Path) -> None:
    path = tmp_path / "rallies.json"
    path.write_text(
        '{"rallies":[{"id":3,"start_frame":100,"end_frame":120,"duration_frames":21,"event_counts":{"bounce":2,"hit":1},"serve_like_start":true,"serve_like_score":0.42,"toss_like_start":true,"toss_rise_px":18.0,"toss_x_span_px":9.0}]}',
        encoding="utf-8",
    )

    rallies = annotate_multimodal_run.load_rallies(path)

    assert rallies == [
        {
            "id": 3,
            "start_frame": 100,
            "end_frame": 120,
            "duration_frames": 21,
            "event_counts": {"bounce": 2, "hit": 1},
            "serve_like_start": True,
            "serve_like_score": 0.42,
            "toss_like_start": True,
            "toss_rise_px": 18.0,
            "toss_x_span_px": 9.0,
        }
    ]
