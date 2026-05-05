from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datapingpong.events.stroke import HitStroke, infer_hit_strokes, infer_stroke_side
from datapingpong.events.trajectory import BallPoint


def test_infer_stroke_side_for_right_handed_players() -> None:
    assert infer_stroke_side("left", "right", racket_hand="right") == "forehand"
    assert infer_stroke_side("left", "left", racket_hand="right") == "backhand"
    assert infer_stroke_side("right", "left", racket_hand="right") == "forehand"
    assert infer_stroke_side("right", "right", racket_hand="right") == "backhand"


def test_infer_hit_strokes_assigns_left_player_forehand() -> None:
    events = [{"event": "hit", "frame": 100}]
    points = [BallPoint(100, 800, 500)]
    pose_features = {
        100: {
            "players": {
                "left": {
                    "pose_detected": 1.0,
                    "shoulder_center_x": 0.35,
                    "shoulder_center_y": 0.45,
                    "shoulder_width": 0.08,
                    "hip_width": 0.08,
                    "torso_length": 0.1,
                    "left_arm_visible": 1.0,
                    "left_wrist_rel_x": -0.2,
                    "left_wrist_rel_y": 0.0,
                    "left_wrist_velocity_x": 0.01,
                    "left_wrist_velocity_y": 0.0,
                    "right_arm_visible": 1.0,
                    "right_wrist_rel_x": 0.07,
                    "right_wrist_rel_y": 0.01,
                    "right_wrist_velocity_x": 0.03,
                    "right_wrist_velocity_y": 0.01,
                },
                "right": {
                    "pose_detected": 1.0,
                    "shoulder_center_x": 0.7,
                    "shoulder_center_y": 0.45,
                    "shoulder_width": 0.08,
                    "hip_width": 0.08,
                    "torso_length": 0.1,
                    "left_arm_visible": 1.0,
                    "left_wrist_rel_x": -0.1,
                    "left_wrist_rel_y": 0.0,
                    "left_wrist_velocity_x": 0.0,
                    "left_wrist_velocity_y": 0.0,
                    "right_arm_visible": 1.0,
                    "right_wrist_rel_x": 0.15,
                    "right_wrist_rel_y": 0.0,
                    "right_wrist_velocity_x": 0.0,
                    "right_wrist_velocity_y": 0.0,
                },
            }
        }
    }

    strokes = infer_hit_strokes(events, points, pose_features, frame_width=1920, frame_height=1080)

    assert len(strokes) == 1
    assert strokes[0].player_role == "left"
    assert strokes[0].striking_arm == "right"
    assert strokes[0].stroke_side == "forehand"


def test_infer_hit_strokes_assigns_right_player_backhand() -> None:
    events = [{"event": "hit", "frame": 100}]
    points = [BallPoint(100, 1500, 500)]
    pose_features = {
        100: {
            "players": {
                "left": {
                    "pose_detected": 1.0,
                    "shoulder_center_x": 0.3,
                    "shoulder_center_y": 0.45,
                    "shoulder_width": 0.08,
                    "hip_width": 0.08,
                    "torso_length": 0.1,
                    "left_arm_visible": 1.0,
                    "left_wrist_rel_x": -0.2,
                    "left_wrist_rel_y": 0.0,
                    "left_wrist_velocity_x": 0.0,
                    "left_wrist_velocity_y": 0.0,
                    "right_arm_visible": 1.0,
                    "right_wrist_rel_x": 0.15,
                    "right_wrist_rel_y": 0.0,
                    "right_wrist_velocity_x": 0.0,
                    "right_wrist_velocity_y": 0.0,
                },
                "right": {
                    "pose_detected": 1.0,
                    "shoulder_center_x": 0.72,
                    "shoulder_center_y": 0.45,
                    "shoulder_width": 0.08,
                    "hip_width": 0.08,
                    "torso_length": 0.1,
                    "left_arm_visible": 1.0,
                    "left_wrist_rel_x": -0.12,
                    "left_wrist_rel_y": 0.0,
                    "left_wrist_velocity_x": 0.01,
                    "left_wrist_velocity_y": 0.01,
                    "right_arm_visible": 1.0,
                    "right_wrist_rel_x": 0.08,
                    "right_wrist_rel_y": 0.0,
                    "right_wrist_velocity_x": 0.02,
                    "right_wrist_velocity_y": 0.01,
                },
            }
        }
    }

    strokes = infer_hit_strokes(events, points, pose_features, frame_width=1920, frame_height=1080)

    assert len(strokes) == 1
    assert strokes[0].player_role == "right"
    assert strokes[0].stroke_side == "backhand"
