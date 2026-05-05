from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datapingpong.pose import PoseFrame, PoseLandmark, build_pose_feature_table, load_multi_pose_frames, load_pose_frames, pose_feature_rows_to_dicts, split_pose_tracks


def test_load_pose_frames_round_trip(tmp_path: Path) -> None:
    frames = [
        PoseFrame(
            frame=3,
            timestamp_ms=100.0,
            detected=True,
            landmarks={"right_wrist": PoseLandmark(0.7, 0.4, -0.1, visibility=0.9, presence=0.9)},
            world_landmarks={"right_wrist": PoseLandmark(0.1, -0.2, -0.3, visibility=0.9, presence=0.9)},
        )
    ]
    path = tmp_path / "pose.json"
    path.write_text(json.dumps({"frames": [{"frame": 3, "timestamp_ms": 100.0, "detected": True, "landmarks": {"right_wrist": {"x": 0.7, "y": 0.4, "z": -0.1, "visibility": 0.9, "presence": 0.9}}, "world_landmarks": {"right_wrist": {"x": 0.1, "y": -0.2, "z": -0.3, "visibility": 0.9, "presence": 0.9}}}]}, indent=2), encoding="utf-8")

    loaded = load_pose_frames(path)

    assert loaded == frames


def test_pose_features_stay_stable_for_scaled_body_geometry() -> None:
    small = [
        _pose_frame(0, wrist_x=0.66, wrist_y=0.50, elbow_x=0.60, elbow_y=0.47),
        _pose_frame(1, wrist_x=0.70, wrist_y=0.44, elbow_x=0.63, elbow_y=0.43),
        _pose_frame(2, wrist_x=0.75, wrist_y=0.40, elbow_x=0.68, elbow_y=0.39),
    ]
    large = [_scale_frame(frame, center=(0.5, 0.55), factor=1.4) for frame in small]

    small_table = build_pose_feature_table(small)
    large_table = build_pose_feature_table(large)
    selected = [
        "right_wrist_rel_x",
        "right_wrist_rel_y",
        "right_elbow_rel_x",
        "right_elbow_rel_y",
        "right_wrist_velocity_x",
        "right_wrist_velocity_y",
        "right_elbow_angle_degrees",
        "right_shoulder_angle_degrees",
    ]
    small_indices = [small_table.feature_names.index(name) for name in selected]
    large_indices = [large_table.feature_names.index(name) for name in selected]

    np.testing.assert_allclose(small_table.features[:, small_indices], large_table.features[:, large_indices], atol=1e-6)


def test_pose_features_zero_fill_missing_arm_landmarks() -> None:
    frame = _pose_frame(0, wrist_x=0.70, wrist_y=0.42, elbow_x=0.62, elbow_y=0.44)
    hidden_wrist = dict(frame.landmarks)
    hidden_wrist["right_wrist"] = PoseLandmark(0.70, 0.42, visibility=0.2)

    table = build_pose_feature_table(
        [
            PoseFrame(
                frame=frame.frame,
                timestamp_ms=frame.timestamp_ms,
                detected=frame.detected,
                landmarks=hidden_wrist,
                world_landmarks=frame.world_landmarks,
            )
        ],
        min_visibility=0.5,
    )
    row = pose_feature_rows_to_dicts(table)[0]

    assert row["right_arm_visible"] == 0.0
    assert row["right_wrist_rel_x"] == 0.0
    assert row["right_wrist_rel_y"] == 0.0
    assert row["right_wrist_above_shoulder"] == 0.0


def test_load_multi_pose_frames_and_split_tracks() -> None:
    path = ROOT / "outputs" / "test_multi_pose_tmp.json"
    path.write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "frame": 10,
                        "timestamp_ms": 100.0,
                        "poses": [
                            {"role": "left", "landmarks": {"right_wrist": {"x": 0.3, "y": 0.6}}},
                            {"role": "right", "landmarks": {"right_wrist": {"x": 0.7, "y": 0.2}}},
                        ],
                    }
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    try:
        frames = load_multi_pose_frames(path)
        tracks = split_pose_tracks(frames)
    finally:
        path.unlink(missing_ok=True)

    assert len(frames) == 1
    assert [pose.role for pose in frames[0].poses] == ["left", "right"]
    assert tracks["left"][0].landmarks["right_wrist"].x == 0.3
    assert tracks["right"][0].landmarks["right_wrist"].x == 0.7


def _pose_frame(
    frame: int,
    *,
    wrist_x: float,
    wrist_y: float,
    elbow_x: float,
    elbow_y: float,
) -> PoseFrame:
    landmarks = {
        "left_shoulder": PoseLandmark(0.42, 0.42, visibility=0.99),
        "right_shoulder": PoseLandmark(0.58, 0.42, visibility=0.99),
        "left_elbow": PoseLandmark(0.38, 0.52, visibility=0.99),
        "right_elbow": PoseLandmark(elbow_x, elbow_y, visibility=0.99),
        "left_wrist": PoseLandmark(0.34, 0.60, visibility=0.99),
        "right_wrist": PoseLandmark(wrist_x, wrist_y, visibility=0.99),
        "left_hip": PoseLandmark(0.45, 0.68, visibility=0.99),
        "right_hip": PoseLandmark(0.55, 0.68, visibility=0.99),
    }
    return PoseFrame(frame=frame, timestamp_ms=frame * 33.3, detected=True, landmarks=landmarks, world_landmarks={})


def _scale_frame(frame: PoseFrame, *, center: tuple[float, float], factor: float) -> PoseFrame:
    scaled_landmarks = {
        name: PoseLandmark(
            x=center[0] + (float(landmark.x) - center[0]) * factor if landmark.x is not None else None,
            y=center[1] + (float(landmark.y) - center[1]) * factor if landmark.y is not None else None,
            z=landmark.z,
            visibility=landmark.visibility,
            presence=landmark.presence,
        )
        for name, landmark in frame.landmarks.items()
    }
    return PoseFrame(
        frame=frame.frame,
        timestamp_ms=frame.timestamp_ms,
        detected=frame.detected,
        landmarks=scaled_landmarks,
        world_landmarks=frame.world_landmarks,
    )
