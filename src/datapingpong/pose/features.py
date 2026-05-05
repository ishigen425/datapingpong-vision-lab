from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .io import PoseFrame, visible_landmark


UPPER_BODY_LANDMARKS = (
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
)


@dataclass(frozen=True)
class PoseFeatureTable:
    frames: list[int]
    timestamps_ms: list[float | None]
    features: np.ndarray
    feature_names: list[str]


def build_pose_feature_table(
    frames: list[PoseFrame],
    *,
    min_visibility: float = 0.5,
    min_presence: float = 0.0,
) -> PoseFeatureTable:
    ordered = sorted(frames, key=lambda frame: frame.frame)
    rows = [_features_for_frame(index, ordered, min_visibility=min_visibility, min_presence=min_presence) for index in range(len(ordered))]
    return PoseFeatureTable(
        frames=[frame.frame for frame in ordered],
        timestamps_ms=[frame.timestamp_ms for frame in ordered],
        features=np.asarray(rows, dtype=np.float64),
        feature_names=_feature_names(),
    )


def pose_feature_rows_to_dicts(table: PoseFeatureTable) -> list[dict[str, float | int | None]]:
    rows: list[dict[str, float | int | None]] = []
    for row_index, frame in enumerate(table.frames):
        row: dict[str, float | int | None] = {"frame": frame, "timestamp_ms": table.timestamps_ms[row_index]}
        for name, value in zip(table.feature_names, table.features[row_index], strict=True):
            row[name] = round(float(value), 6)
        rows.append(row)
    return rows


def _features_for_frame(
    index: int,
    frames: list[PoseFrame],
    *,
    min_visibility: float,
    min_presence: float,
) -> list[float]:
    frame = frames[index]
    shoulder_center = _midpoint(frame, "left_shoulder", "right_shoulder", min_visibility=min_visibility, min_presence=min_presence)
    hip_center = _midpoint(frame, "left_hip", "right_hip", min_visibility=min_visibility, min_presence=min_presence)
    shoulder_width = _pair_distance(frame, "left_shoulder", "right_shoulder", min_visibility=min_visibility, min_presence=min_presence)
    hip_width = _pair_distance(frame, "left_hip", "right_hip", min_visibility=min_visibility, min_presence=min_presence)
    torso_length = _distance_between_points(shoulder_center, hip_center)
    body_scale = max(shoulder_width, torso_length, hip_width, 1e-6)

    values: list[float] = [
        1.0 if frame.detected else 0.0,
        _visible_ratio(frame, min_visibility=min_visibility, min_presence=min_presence),
        shoulder_center[0] if shoulder_center is not None else 0.0,
        shoulder_center[1] if shoulder_center is not None else 0.0,
        hip_center[0] if hip_center is not None else 0.0,
        hip_center[1] if hip_center is not None else 0.0,
        shoulder_width,
        hip_width,
        torso_length,
        _line_angle_degrees(frame, "left_shoulder", "right_shoulder", min_visibility=min_visibility, min_presence=min_presence),
        _line_angle_degrees(frame, "left_hip", "right_hip", min_visibility=min_visibility, min_presence=min_presence),
        _tilt_from_points(shoulder_center, hip_center),
    ]

    for side in ("left", "right"):
        wrist = visible_landmark(frame, f"{side}_wrist", min_visibility=min_visibility, min_presence=min_presence)
        elbow = visible_landmark(frame, f"{side}_elbow", min_visibility=min_visibility, min_presence=min_presence)
        shoulder = visible_landmark(frame, f"{side}_shoulder", min_visibility=min_visibility, min_presence=min_presence)
        hip = visible_landmark(frame, f"{side}_hip", min_visibility=min_visibility, min_presence=min_presence)

        arm_visible = 1.0 if wrist is not None and elbow is not None and shoulder is not None else 0.0
        wrist_rel = _relative_to_center(wrist, shoulder_center, body_scale)
        elbow_rel = _relative_to_center(elbow, shoulder_center, body_scale)
        wrist_velocity = _relative_velocity(
            index,
            frames,
            f"{side}_wrist",
            min_visibility=min_visibility,
            min_presence=min_presence,
        )

        values.extend(
            [
                arm_visible,
                wrist_rel[0],
                wrist_rel[1],
                elbow_rel[0],
                elbow_rel[1],
                wrist_velocity[0] / body_scale,
                wrist_velocity[1] / body_scale,
                _pair_distance(frame, f"{side}_wrist", f"{side}_elbow", min_visibility=min_visibility, min_presence=min_presence) / body_scale,
                _joint_angle(frame, f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist", min_visibility=min_visibility, min_presence=min_presence),
                _joint_angle(frame, f"{side}_elbow", f"{side}_shoulder", f"{side}_hip", min_visibility=min_visibility, min_presence=min_presence),
                1.0 if wrist is not None and shoulder is not None and float(wrist.y) < float(shoulder.y) else 0.0,
            ]
        )
    return values


def _visible_ratio(frame: PoseFrame, *, min_visibility: float, min_presence: float) -> float:
    visible = sum(
        1
        for name in UPPER_BODY_LANDMARKS
        if visible_landmark(frame, name, min_visibility=min_visibility, min_presence=min_presence) is not None
    )
    return visible / len(UPPER_BODY_LANDMARKS)


def _midpoint(
    frame: PoseFrame,
    first: str,
    second: str,
    *,
    min_visibility: float,
    min_presence: float,
) -> tuple[float, float] | None:
    first_landmark = visible_landmark(frame, first, min_visibility=min_visibility, min_presence=min_presence)
    second_landmark = visible_landmark(frame, second, min_visibility=min_visibility, min_presence=min_presence)
    if first_landmark is None or second_landmark is None:
        return None
    return ((float(first_landmark.x) + float(second_landmark.x)) / 2.0, (float(first_landmark.y) + float(second_landmark.y)) / 2.0)


def _pair_distance(
    frame: PoseFrame,
    first: str,
    second: str,
    *,
    min_visibility: float,
    min_presence: float,
) -> float:
    first_landmark = visible_landmark(frame, first, min_visibility=min_visibility, min_presence=min_presence)
    second_landmark = visible_landmark(frame, second, min_visibility=min_visibility, min_presence=min_presence)
    if first_landmark is None or second_landmark is None:
        return 0.0
    return math.hypot(float(second_landmark.x) - float(first_landmark.x), float(second_landmark.y) - float(first_landmark.y))


def _distance_between_points(first: tuple[float, float] | None, second: tuple[float, float] | None) -> float:
    if first is None or second is None:
        return 0.0
    return math.hypot(second[0] - first[0], second[1] - first[1])


def _relative_to_center(
    landmark: object | None,
    center: tuple[float, float] | None,
    body_scale: float,
) -> tuple[float, float]:
    if landmark is None or center is None:
        return 0.0, 0.0
    return ((float(landmark.x) - center[0]) / body_scale, (float(landmark.y) - center[1]) / body_scale)


def _relative_velocity(
    index: int,
    frames: list[PoseFrame],
    name: str,
    *,
    min_visibility: float,
    min_presence: float,
) -> tuple[float, float]:
    current = visible_landmark(frames[index], name, min_visibility=min_visibility, min_presence=min_presence)
    if current is None:
        return 0.0, 0.0
    previous = _neighbor(index, frames, name, step=-1, min_visibility=min_visibility, min_presence=min_presence)
    following = _neighbor(index, frames, name, step=1, min_visibility=min_visibility, min_presence=min_presence)
    if previous is not None and following is not None:
        previous_frame, previous_landmark = previous
        following_frame, following_landmark = following
        frame_delta = max(1, following_frame.frame - previous_frame.frame)
        return (
            (float(following_landmark.x) - float(previous_landmark.x)) / frame_delta,
            (float(following_landmark.y) - float(previous_landmark.y)) / frame_delta,
        )
    if previous is not None:
        previous_frame, previous_landmark = previous
        frame_delta = max(1, frames[index].frame - previous_frame.frame)
        return (
            (float(current.x) - float(previous_landmark.x)) / frame_delta,
            (float(current.y) - float(previous_landmark.y)) / frame_delta,
        )
    if following is not None:
        following_frame, following_landmark = following
        frame_delta = max(1, following_frame.frame - frames[index].frame)
        return (
            (float(following_landmark.x) - float(current.x)) / frame_delta,
            (float(following_landmark.y) - float(current.y)) / frame_delta,
        )
    return 0.0, 0.0


def _neighbor(
    index: int,
    frames: list[PoseFrame],
    name: str,
    *,
    step: int,
    min_visibility: float,
    min_presence: float,
) -> tuple[PoseFrame, object] | None:
    cursor = index + step
    while 0 <= cursor < len(frames):
        landmark = visible_landmark(frames[cursor], name, min_visibility=min_visibility, min_presence=min_presence)
        if landmark is not None:
            return frames[cursor], landmark
        cursor += step
    return None


def _joint_angle(
    frame: PoseFrame,
    first: str,
    joint: str,
    second: str,
    *,
    min_visibility: float,
    min_presence: float,
) -> float:
    first_landmark = visible_landmark(frame, first, min_visibility=min_visibility, min_presence=min_presence)
    joint_landmark = visible_landmark(frame, joint, min_visibility=min_visibility, min_presence=min_presence)
    second_landmark = visible_landmark(frame, second, min_visibility=min_visibility, min_presence=min_presence)
    if first_landmark is None or joint_landmark is None or second_landmark is None:
        return 0.0
    return _angle_between_points(
        (float(first_landmark.x), float(first_landmark.y)),
        (float(joint_landmark.x), float(joint_landmark.y)),
        (float(second_landmark.x), float(second_landmark.y)),
    )


def _angle_between_points(
    first: tuple[float, float],
    joint: tuple[float, float],
    second: tuple[float, float],
) -> float:
    first_vector = (first[0] - joint[0], first[1] - joint[1])
    second_vector = (second[0] - joint[0], second[1] - joint[1])
    first_norm = math.hypot(*first_vector)
    second_norm = math.hypot(*second_vector)
    if first_norm == 0.0 or second_norm == 0.0:
        return 0.0
    cosine = max(-1.0, min(1.0, (first_vector[0] * second_vector[0] + first_vector[1] * second_vector[1]) / (first_norm * second_norm)))
    return math.degrees(math.acos(cosine))


def _line_angle_degrees(
    frame: PoseFrame,
    first: str,
    second: str,
    *,
    min_visibility: float,
    min_presence: float,
) -> float:
    first_landmark = visible_landmark(frame, first, min_visibility=min_visibility, min_presence=min_presence)
    second_landmark = visible_landmark(frame, second, min_visibility=min_visibility, min_presence=min_presence)
    if first_landmark is None or second_landmark is None:
        return 0.0
    return math.degrees(math.atan2(float(second_landmark.y) - float(first_landmark.y), float(second_landmark.x) - float(first_landmark.x)))


def _tilt_from_points(first: tuple[float, float] | None, second: tuple[float, float] | None) -> float:
    if first is None or second is None:
        return 0.0
    return math.degrees(math.atan2(second[1] - first[1], second[0] - first[0]))


def _feature_names() -> list[str]:
    names = [
        "pose_detected",
        "upper_body_visible_ratio",
        "shoulder_center_x",
        "shoulder_center_y",
        "hip_center_x",
        "hip_center_y",
        "shoulder_width",
        "hip_width",
        "torso_length",
        "shoulder_tilt_degrees",
        "hip_tilt_degrees",
        "torso_tilt_degrees",
    ]
    for side in ("left", "right"):
        names.extend(
            [
                f"{side}_arm_visible",
                f"{side}_wrist_rel_x",
                f"{side}_wrist_rel_y",
                f"{side}_elbow_rel_x",
                f"{side}_elbow_rel_y",
                f"{side}_wrist_velocity_x",
                f"{side}_wrist_velocity_y",
                f"{side}_wrist_to_elbow_distance",
                f"{side}_elbow_angle_degrees",
                f"{side}_shoulder_angle_degrees",
                f"{side}_wrist_above_shoulder",
            ]
        )
    return names
