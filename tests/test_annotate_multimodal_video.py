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
