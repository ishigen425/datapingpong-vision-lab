from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
import math

import numpy as np

from .trajectory import BallPoint, score_trajectory


@dataclass(frozen=True)
class FeatureTable:
    item: str
    frames: list[int]
    features: np.ndarray
    feature_names: list[str]


def build_feature_table(
    item: str,
    points: list[BallPoint],
    *,
    offsets: tuple[int, ...] = (-6, -4, -2, 0, 2, 4, 6),
    smooth_window: int = 5,
    max_gap: int = 3,
) -> FeatureTable:
    ordered = sorted(points, key=lambda point: point.frame)
    frames = [point.frame for point in ordered]
    by_frame = {point.frame: point for point in ordered}
    probability_rows = {row.frame: row for row in score_trajectory(ordered, smooth_window=smooth_window, max_gap=max_gap)}
    feature_names = _feature_names(offsets)
    rows = []
    for frame in frames:
        point = by_frame[frame]
        if not point.detected:
            continue
        rows.append(_features_for_frame(frame, by_frame, frames, probability_rows, offsets))
    valid_frames = [frame for frame in frames if by_frame[frame].detected]
    return FeatureTable(item=item, frames=valid_frames, features=np.asarray(rows, dtype=np.float64), feature_names=feature_names)


def _features_for_frame(
    frame: int,
    by_frame: dict[int, BallPoint],
    frames: list[int],
    probability_rows: dict[int, object],
    offsets: tuple[int, ...],
) -> list[float]:
    center = by_frame[frame]
    center_x = float(center.x or 0.0)
    center_y = float(center.y or 0.0)
    values: list[float] = []

    detected_count = 0
    for offset in offsets:
        point = by_frame.get(frame + offset)
        if point is None or not point.detected or point.x is None or point.y is None:
            values.extend([0.0, 0.0, 1.0])
            continue
        detected_count += 1
        values.extend([float(point.x) - center_x, float(point.y) - center_y, 0.0])

    values.append(detected_count / len(offsets))
    values.extend(_velocity_features(frame, by_frame, frames))
    row = probability_rows.get(frame)
    values.extend(
        [
            float(getattr(row, "speed", 0.0) or 0.0),
            float(getattr(row, "acceleration", 0.0) or 0.0),
            float(getattr(row, "turn_angle_degrees", 0.0) or 0.0),
        ]
    )
    values.extend([center_x / 1280.0, center_y / 720.0])
    return values


def _velocity_features(frame: int, by_frame: dict[int, BallPoint], frames: list[int]) -> list[float]:
    index = bisect_left(frames, frame)
    previous_point = _neighbor(index, frames, by_frame, -1)
    next_point = _neighbor(index, frames, by_frame, 1)
    before_previous = _neighbor(index, frames, by_frame, -2)
    after_next = _neighbor(index, frames, by_frame, 2)

    vx_before, vy_before = _delta(previous_point, by_frame[frame])
    vx_after, vy_after = _delta(by_frame[frame], next_point)
    vx_wide_before, vy_wide_before = _delta(before_previous, by_frame[frame])
    vx_wide_after, vy_wide_after = _delta(by_frame[frame], after_next)
    speed_before = math.hypot(vx_before, vy_before)
    speed_after = math.hypot(vx_after, vy_after)
    y_flip = 1.0 if vy_before > 0.0 and vy_after < 0.0 else 0.0
    return [
        vx_before,
        vy_before,
        vx_after,
        vy_after,
        vx_wide_before,
        vy_wide_before,
        vx_wide_after,
        vy_wide_after,
        speed_before,
        speed_after,
        speed_after - speed_before,
        y_flip,
    ]


def _neighbor(index: int, frames: list[int], by_frame: dict[int, BallPoint], offset: int) -> BallPoint | None:
    target = index + offset
    if target < 0 or target >= len(frames):
        return None
    point = by_frame[frames[target]]
    if not point.detected:
        return None
    return point


def _delta(start: BallPoint | None, end: BallPoint | None) -> tuple[float, float]:
    if start is None or end is None or start.x is None or start.y is None or end.x is None or end.y is None:
        return 0.0, 0.0
    frame_delta = max(1, end.frame - start.frame)
    return (float(end.x) - float(start.x)) / frame_delta, (float(end.y) - float(start.y)) / frame_delta


def _feature_names(offsets: tuple[int, ...]) -> list[str]:
    names: list[str] = []
    for offset in offsets:
        names.extend([f"dx_{offset:+d}", f"dy_{offset:+d}", f"missing_{offset:+d}"])
    names.extend(
        [
            "detected_ratio",
            "vx_before",
            "vy_before",
            "vx_after",
            "vy_after",
            "vx_wide_before",
            "vy_wide_before",
            "vx_wide_after",
            "vy_wide_after",
            "speed_before",
            "speed_after",
            "speed_delta",
            "y_flip",
            "rule_speed",
            "rule_acceleration",
            "rule_turn_angle_degrees",
            "x_norm",
            "y_norm",
        ]
    )
    return names
