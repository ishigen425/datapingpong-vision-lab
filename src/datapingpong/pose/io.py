from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any


POSE_LANDMARK_NAMES = (
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
)

DEFAULT_POSE_ROLES = ("left", "right")


@dataclass(frozen=True)
class PoseLandmark:
    x: float | None
    y: float | None
    z: float | None = None
    visibility: float | None = None
    presence: float | None = None


@dataclass(frozen=True)
class PoseFrame:
    frame: int
    timestamp_ms: float | None
    detected: bool
    landmarks: dict[str, PoseLandmark]
    world_landmarks: dict[str, PoseLandmark]


@dataclass(frozen=True)
class PoseInstance:
    role: str
    landmarks: dict[str, PoseLandmark]
    world_landmarks: dict[str, PoseLandmark]


@dataclass(frozen=True)
class MultiPoseFrame:
    frame: int
    timestamp_ms: float | None
    poses: list[PoseInstance]

    @property
    def detected(self) -> bool:
        return bool(self.poses)


def landmark_names() -> list[str]:
    return list(POSE_LANDMARK_NAMES)


def load_pose_frames(path: Path) -> list[PoseFrame]:
    frames = load_multi_pose_frames(path)
    return [_primary_pose_frame(frame) for frame in frames]


def load_multi_pose_frames(path: Path) -> list[MultiPoseFrame]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["frames"] if isinstance(payload, dict) and "frames" in payload else payload
    if not isinstance(rows, list):
        raise ValueError(f"Unsupported pose input format: {path}")
    frames = [_row_to_multi_pose_frame(row, index) for index, row in enumerate(rows)]
    return sorted(frames, key=lambda frame: frame.frame)


def pose_frames_to_dicts(frames: list[PoseFrame]) -> list[dict[str, Any]]:
    return [_pose_frame_to_dict(frame) for frame in frames]


def multi_pose_frames_to_dicts(frames: list[MultiPoseFrame]) -> list[dict[str, Any]]:
    return [_multi_pose_frame_to_dict(frame) for frame in frames]


def split_pose_tracks(
    frames: list[MultiPoseFrame],
    *,
    roles: tuple[str, ...] = DEFAULT_POSE_ROLES,
) -> dict[str, list[PoseFrame]]:
    tracks: dict[str, list[PoseFrame]] = {role: [] for role in roles}
    for frame in sorted(frames, key=lambda row: row.frame):
        by_role = {pose.role: pose for pose in frame.poses}
        for role in roles:
            pose = by_role.get(role)
            tracks[role].append(
                PoseFrame(
                    frame=frame.frame,
                    timestamp_ms=frame.timestamp_ms,
                    detected=pose is not None and bool(pose.landmarks),
                    landmarks={} if pose is None else pose.landmarks,
                    world_landmarks={} if pose is None else pose.world_landmarks,
                )
            )
    return tracks


def assign_pose_roles(instances: list[PoseInstance]) -> list[PoseInstance]:
    ordered = sorted(instances, key=_pose_sort_key)
    roles = list(DEFAULT_POSE_ROLES)
    labeled: list[PoseInstance] = []
    for index, instance in enumerate(ordered):
        role = roles[index] if index < len(roles) else f"extra_{index - len(roles) + 1}"
        labeled.append(replace(instance, role=role))
    return labeled


def visible_landmark(
    frame_or_pose: PoseFrame | PoseInstance,
    name: str,
    *,
    min_visibility: float = 0.0,
    min_presence: float = 0.0,
    world: bool = False,
) -> PoseLandmark | None:
    landmarks = frame_or_pose.world_landmarks if world else frame_or_pose.landmarks
    landmark = landmarks.get(name)
    if landmark is None or landmark.x is None or landmark.y is None:
        return None
    if landmark.visibility is not None and landmark.visibility < min_visibility:
        return None
    if landmark.presence is not None and landmark.presence < min_presence:
        return None
    return landmark


def _primary_pose_frame(frame: MultiPoseFrame) -> PoseFrame:
    if not frame.poses:
        return PoseFrame(frame=frame.frame, timestamp_ms=frame.timestamp_ms, detected=False, landmarks={}, world_landmarks={})
    preferred = next((pose for pose in frame.poses if pose.role == "left"), frame.poses[0])
    return PoseFrame(
        frame=frame.frame,
        timestamp_ms=frame.timestamp_ms,
        detected=bool(preferred.landmarks),
        landmarks=preferred.landmarks,
        world_landmarks=preferred.world_landmarks,
    )


def _multi_pose_frame_to_dict(frame: MultiPoseFrame) -> dict[str, Any]:
    return {
        "frame": frame.frame,
        "timestamp_ms": frame.timestamp_ms,
        "pose_count": len(frame.poses),
        "poses": [
            {
                "role": pose.role,
                "landmarks": _landmarks_to_dict(pose.landmarks),
                "world_landmarks": _landmarks_to_dict(pose.world_landmarks),
            }
            for pose in frame.poses
        ],
    }


def _pose_frame_to_dict(frame: PoseFrame) -> dict[str, Any]:
    return {
        "frame": frame.frame,
        "timestamp_ms": frame.timestamp_ms,
        "detected": frame.detected,
        "landmarks": _landmarks_to_dict(frame.landmarks),
        "world_landmarks": _landmarks_to_dict(frame.world_landmarks),
    }


def _landmarks_to_dict(landmarks: dict[str, PoseLandmark]) -> dict[str, dict[str, float | None]]:
    return {
        name: {
            "x": landmark.x,
            "y": landmark.y,
            "z": landmark.z,
            "visibility": landmark.visibility,
            "presence": landmark.presence,
        }
        for name, landmark in landmarks.items()
    }


def _row_to_multi_pose_frame(row: Any, index: int) -> MultiPoseFrame:
    if not isinstance(row, dict):
        raise ValueError(f"Expected pose frame mapping at index {index}, got {type(row).__name__}")
    if isinstance(row.get("poses"), list):
        poses = [_row_to_pose_instance(pose_row, pose_index) for pose_index, pose_row in enumerate(row["poses"])]
        return MultiPoseFrame(
            frame=int(row.get("frame", index)),
            timestamp_ms=float(row["timestamp_ms"]) if row.get("timestamp_ms") is not None else None,
            poses=poses,
        )
    return MultiPoseFrame(
        frame=int(row.get("frame", index)),
        timestamp_ms=float(row["timestamp_ms"]) if row.get("timestamp_ms") is not None else None,
        poses=[
            PoseInstance(
                role=str(row.get("role", "near")),
                landmarks=_mapping_to_landmarks(row.get("landmarks", {})),
                world_landmarks=_mapping_to_landmarks(row.get("world_landmarks", {})),
            )
        ]
        if row.get("landmarks") or row.get("world_landmarks")
        else [],
    )


def _row_to_pose_instance(row: Any, index: int) -> PoseInstance:
    if not isinstance(row, dict):
        raise ValueError(f"Expected pose mapping at index {index}, got {type(row).__name__}")
    return PoseInstance(
        role=str(row.get("role", f"pose_{index}")),
        landmarks=_mapping_to_landmarks(row.get("landmarks", {})),
        world_landmarks=_mapping_to_landmarks(row.get("world_landmarks", {})),
    )


def _mapping_to_landmarks(payload: Any) -> dict[str, PoseLandmark]:
    if not isinstance(payload, dict):
        return {}
    landmarks: dict[str, PoseLandmark] = {}
    for name, value in payload.items():
        if not isinstance(value, dict):
            continue
        landmarks[str(name)] = PoseLandmark(
            x=float(value["x"]) if value.get("x") is not None else None,
            y=float(value["y"]) if value.get("y") is not None else None,
            z=float(value["z"]) if value.get("z") is not None else None,
            visibility=float(value["visibility"]) if value.get("visibility") is not None else None,
            presence=float(value["presence"]) if value.get("presence") is not None else None,
        )
    return landmarks


def _pose_sort_key(pose: PoseInstance) -> float:
    center = _pose_center_x(pose)
    if center is not None:
        return center
    return 1e9


def _pose_center_x(pose: PoseInstance) -> float | None:
    x_values = [
        float(landmark.x)
        for name, landmark in pose.landmarks.items()
        if name in {"left_shoulder", "right_shoulder", "left_hip", "right_hip"} and landmark.x is not None
    ]
    if not x_values:
        return None
    return sum(x_values) / len(x_values)
