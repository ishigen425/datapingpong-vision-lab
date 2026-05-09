from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import median


@dataclass(frozen=True)
class BallPoint:
    frame: int
    x: float | None
    y: float | None
    confidence: float = 1.0
    detected: bool = True


@dataclass(frozen=True)
class EventProbabilities:
    frame: int
    x: float | None
    y: float | None
    detected: bool
    confidence: float
    bounce_probability: float
    hit_probability: float
    speed: float | None
    acceleration: float | None
    turn_angle_degrees: float | None
    poly_bounce_probability: float | None = None


@dataclass(frozen=True)
class EventPeak:
    event: str
    frame: int
    probability: float
    x: float | None
    y: float | None
    local_x_span: float | None = None
    local_y_span: float | None = None
    local_detections: int | None = None
    local_before_x_displacement: float | None = None
    local_after_x_displacement: float | None = None


def score_trajectory(
    points: list[BallPoint],
    *,
    max_gap: int = 3,
    smooth_window: int = 5,
    polynomial_bounce_weight: float = 0.0,
) -> list[EventProbabilities]:
    """Estimate bounce and hit probabilities from a tracked ball trajectory.

    The detector intentionally uses only ball coordinates. It scores short-term
    kinematic cues that are useful before committing to a trainable event model:
    y-velocity sign flips for bounces, vector-angle changes for hits, and local
    speed/acceleration spikes for both.
    """

    if not points:
        return []

    ordered = sorted(points, key=lambda point: point.frame)
    frames = [point.frame for point in ordered]
    raw_x = [point.x if point.detected else None for point in ordered]
    raw_y = [point.y if point.detected else None for point in ordered]
    confidences = [max(0.0, min(1.0, point.confidence)) for point in ordered]
    detected = [point.detected and point.x is not None and point.y is not None for point in ordered]

    x_values = _smooth(_interpolate_short_gaps(raw_x, max_gap), smooth_window)
    y_values = _smooth(_interpolate_short_gaps(raw_y, max_gap), smooth_window)

    velocities: list[tuple[float, float] | None] = []
    speeds: list[float | None] = []
    for index in range(len(ordered)):
        velocity = _central_velocity(frames, x_values, y_values, index)
        velocities.append(velocity)
        speeds.append(math.hypot(velocity[0], velocity[1]) if velocity else None)

    accelerations: list[float | None] = []
    turn_angles: list[float | None] = []
    for index in range(len(ordered)):
        accelerations.append(_acceleration(speeds, index))
        turn_angles.append(_turn_angle(velocities, index))

    speed_scale = _robust_scale([speed for speed in speeds if speed is not None])
    accel_scale = _robust_scale([accel for accel in accelerations if accel is not None])
    poly_bounce_scores = _polynomial_bounce_scores(frames, x_values, y_values, detected, confidences)

    rows: list[EventProbabilities] = []
    for index, point in enumerate(ordered):
        track_quality = _track_quality(detected, confidences, index)
        y_flip = _y_velocity_flip_score(velocities, index)
        x_flip = _x_velocity_flip_score(velocities, index, speed_scale)
        x_delta = _x_velocity_delta_score(velocities, index, speed_scale)
        angle_score = _ratio((turn_angles[index] or 0.0), 70.0)
        accel_score = _ratio((accelerations[index] or 0.0), accel_scale * 2.0)
        speed_score = _ratio((speeds[index] or 0.0), speed_scale * 1.5)
        missing_edge = _missing_edge_score(detected, index)

        kinematic_bounce = _clamp01(
            track_quality
            * (
                0.58 * y_flip
                + 0.18 * accel_score
                + 0.16 * angle_score
                + 0.08 * speed_score
            )
        )
        poly_bounce = poly_bounce_scores[index]
        bounce = max(kinematic_bounce, _clamp01(0.72 * kinematic_bounce + polynomial_bounce_weight * poly_bounce))
        hit = _clamp01(
            track_quality
            * (
                0.46 * x_flip
                + 0.22 * x_delta
                + 0.16 * angle_score
                + 0.10 * accel_score
                + 0.04 * speed_score
                + 0.02 * missing_edge
            )
        )

        rows.append(
            EventProbabilities(
                frame=point.frame,
                x=point.x,
                y=point.y,
                detected=detected[index],
                confidence=confidences[index],
                bounce_probability=round(bounce, 6),
                hit_probability=round(hit, 6),
                speed=round(speeds[index], 6) if speeds[index] is not None else None,
                acceleration=round(accelerations[index], 6) if accelerations[index] is not None else None,
                turn_angle_degrees=round(turn_angles[index], 6) if turn_angles[index] is not None else None,
                poly_bounce_probability=round(poly_bounce, 6),
            )
        )
    return rows


def detect_event_peaks(
    probabilities: list[EventProbabilities],
    *,
    bounce_threshold: float = 0.65,
    hit_threshold: float = 0.70,
    nms_window: int = 8,
) -> list[EventPeak]:
    candidates: list[EventPeak] = []
    for row in probabilities:
        if row.bounce_probability >= bounce_threshold:
            candidates.append(EventPeak("bounce", row.frame, row.bounce_probability, row.x, row.y))
        if row.hit_probability >= hit_threshold:
            candidates.append(EventPeak("hit", row.frame, row.hit_probability, row.x, row.y))

    selected: list[EventPeak] = []
    for event in ("bounce", "hit"):
        event_candidates = sorted(
            [candidate for candidate in candidates if candidate.event == event],
            key=lambda candidate: candidate.probability,
            reverse=True,
        )
        kept: list[EventPeak] = []
        for candidate in event_candidates:
            if all(abs(candidate.frame - existing.frame) > nms_window for existing in kept):
                kept.append(candidate)
        selected.extend(kept)
    return sorted(selected, key=lambda peak: (peak.frame, peak.event))


def _interpolate_short_gaps(values: list[float | None], max_gap: int) -> list[float | None]:
    result = values[:]
    index = 0
    while index < len(result):
        if result[index] is not None:
            index += 1
            continue
        start = index
        while index < len(result) and result[index] is None:
            index += 1
        end = index
        gap = end - start
        left = start - 1
        right = end
        if gap <= max_gap and left >= 0 and right < len(result) and result[left] is not None and result[right] is not None:
            left_value = float(result[left])
            right_value = float(result[right])
            for offset in range(1, gap + 1):
                result[start + offset - 1] = left_value + (right_value - left_value) * offset / (gap + 1)
    return result


def _smooth(values: list[float | None], window: int) -> list[float | None]:
    if window <= 1:
        return values
    radius = window // 2
    result: list[float | None] = []
    for index in range(len(values)):
        local = [
            float(value)
            for value in values[max(0, index - radius) : min(len(values), index + radius + 1)]
            if value is not None
        ]
        result.append(median(local) if local else None)
    return result


def _central_velocity(
    frames: list[int],
    x_values: list[float | None],
    y_values: list[float | None],
    index: int,
) -> tuple[float, float] | None:
    if index <= 0 or index >= len(frames) - 1:
        return None
    if x_values[index - 1] is None or x_values[index + 1] is None:
        return None
    if y_values[index - 1] is None or y_values[index + 1] is None:
        return None
    frame_delta = frames[index + 1] - frames[index - 1]
    if frame_delta <= 0:
        return None
    return (
        (float(x_values[index + 1]) - float(x_values[index - 1])) / frame_delta,
        (float(y_values[index + 1]) - float(y_values[index - 1])) / frame_delta,
    )


def _acceleration(speeds: list[float | None], index: int) -> float | None:
    if index <= 0 or index >= len(speeds) - 1:
        return None
    if speeds[index - 1] is None or speeds[index] is None or speeds[index + 1] is None:
        return None
    return abs(float(speeds[index + 1]) - 2.0 * float(speeds[index]) + float(speeds[index - 1]))


def _turn_angle(velocities: list[tuple[float, float] | None], index: int) -> float | None:
    if index <= 0 or index >= len(velocities) - 1:
        return None
    before = velocities[index - 1]
    after = velocities[index + 1]
    if before is None or after is None:
        return None
    before_norm = math.hypot(before[0], before[1])
    after_norm = math.hypot(after[0], after[1])
    if before_norm == 0.0 or after_norm == 0.0:
        return None
    cosine = _clamp((before[0] * after[0] + before[1] * after[1]) / (before_norm * after_norm), -1.0, 1.0)
    return math.degrees(math.acos(cosine))


def _y_velocity_flip_score(velocities: list[tuple[float, float] | None], index: int) -> float:
    if index <= 0 or index >= len(velocities) - 1:
        return 0.0
    before = velocities[index - 1]
    after = velocities[index + 1]
    if before is None or after is None:
        return 0.0
    before_y = before[1]
    after_y = after[1]
    if before_y <= 0.0 or after_y >= 0.0:
        return 0.0
    strength = min(abs(before_y), abs(after_y))
    return _ratio(strength, 1.2)


def _x_velocity_flip_score(velocities: list[tuple[float, float] | None], index: int, speed_scale: float) -> float:
    if index <= 0 or index >= len(velocities) - 1:
        return 0.0
    before = velocities[index - 1]
    after = velocities[index + 1]
    if before is None or after is None:
        return 0.0
    before_x = before[0]
    after_x = after[0]
    if before_x == 0.0 or after_x == 0.0 or before_x * after_x >= 0.0:
        return 0.0
    strength = min(abs(before_x), abs(after_x))
    return _ratio(strength, speed_scale * 0.55)


def _x_velocity_delta_score(velocities: list[tuple[float, float] | None], index: int, speed_scale: float) -> float:
    if index <= 0 or index >= len(velocities) - 1:
        return 0.0
    before = velocities[index - 1]
    after = velocities[index + 1]
    if before is None or after is None:
        return 0.0
    return _ratio(abs(after[0] - before[0]), speed_scale * 1.2)


def _polynomial_bounce_scores(
    frames: list[int],
    x_values: list[float | None],
    y_values: list[float | None],
    detected: list[bool],
    confidences: list[float],
    *,
    fit_radius: int = 4,
) -> list[float]:
    scores = [0.0] * len(frames)
    if len(frames) < fit_radius * 2 + 1:
        return scores

    step_distances = []
    for index in range(1, len(frames)):
        if x_values[index - 1] is None or x_values[index] is None:
            continue
        if y_values[index - 1] is None or y_values[index] is None:
            continue
        step_distances.append(
            math.hypot(float(x_values[index]) - float(x_values[index - 1]), float(y_values[index]) - float(y_values[index - 1]))
        )
    distance_scale = max(_robust_scale(step_distances) * 0.75, 2.0)

    for index, frame in enumerate(frames):
        if not detected[index] or x_values[index] is None or y_values[index] is None:
            continue
        left_indices = [
            candidate
            for candidate in range(max(0, index - fit_radius), index)
            if detected[candidate] and x_values[candidate] is not None and y_values[candidate] is not None
        ]
        right_indices = [
            candidate
            for candidate in range(index + 1, min(len(frames), index + fit_radius + 1))
            if detected[candidate] and x_values[candidate] is not None and y_values[candidate] is not None
        ]
        if len(left_indices) < 3 or len(right_indices) < 3:
            continue

        left_x = _fit_quadratic([(frames[candidate] - frame, float(x_values[candidate])) for candidate in left_indices])
        left_y = _fit_quadratic([(frames[candidate] - frame, float(y_values[candidate])) for candidate in left_indices])
        right_x = _fit_quadratic([(frames[candidate] - frame, float(x_values[candidate])) for candidate in right_indices])
        right_y = _fit_quadratic([(frames[candidate] - frame, float(y_values[candidate])) for candidate in right_indices])
        if left_x is None or left_y is None or right_x is None or right_y is None:
            continue

        best_t = 0.0
        best_distance = float("inf")
        for step in range(-10, 11):
            t = step / 10.0
            dx = _eval_quadratic(left_x, t) - _eval_quadratic(right_x, t)
            dy = _eval_quadratic(left_y, t) - _eval_quadratic(right_y, t)
            distance = math.hypot(dx, dy)
            if distance < best_distance:
                best_t = t
                best_distance = distance

        left_dy = _eval_quadratic_derivative(left_y, best_t)
        right_dy = _eval_quadratic_derivative(right_y, best_t)
        if left_dy <= 0.0 or right_dy >= 0.0:
            continue

        intersection_x = (_eval_quadratic(left_x, best_t) + _eval_quadratic(right_x, best_t)) / 2.0
        intersection_y = (_eval_quadratic(left_y, best_t) + _eval_quadratic(right_y, best_t)) / 2.0
        center_distance = math.hypot(intersection_x - float(x_values[index]), intersection_y - float(y_values[index]))
        velocity_scale = max(distance_scale, 1.0)
        y_flip_score = _ratio(min(abs(left_dy), abs(right_dy)), velocity_scale)
        intersection_score = 1.0 - _ratio(best_distance, distance_scale * 2.0)
        center_score = 1.0 - _ratio(center_distance, distance_scale * 2.5)
        time_score = 1.0 - _ratio(abs(best_t), 1.0)
        scores[index] = _clamp01(
            _track_quality(detected, confidences, index, radius=fit_radius)
            * (0.42 * y_flip_score + 0.26 * intersection_score + 0.22 * center_score + 0.10 * time_score)
        )
    return scores


def _fit_quadratic(points: list[tuple[float, float]]) -> tuple[float, float, float] | None:
    if len(points) < 3:
        return None
    s0 = float(len(points))
    s1 = sum(t for t, _ in points)
    s2 = sum(t * t for t, _ in points)
    s3 = sum(t * t * t for t, _ in points)
    s4 = sum(t * t * t * t for t, _ in points)
    v0 = sum(value for _, value in points)
    v1 = sum(t * value for t, value in points)
    v2 = sum(t * t * value for t, value in points)
    return _solve_3x3(
        [
            [s4, s3, s2],
            [s3, s2, s1],
            [s2, s1, s0],
        ],
        [v2, v1, v0],
    )


def _solve_3x3(matrix: list[list[float]], vector: list[float]) -> tuple[float, float, float] | None:
    rows = [matrix[row][:] + [vector[row]] for row in range(3)]
    for column in range(3):
        pivot = max(range(column, 3), key=lambda row: abs(rows[row][column]))
        if abs(rows[pivot][column]) < 1e-9:
            return None
        rows[column], rows[pivot] = rows[pivot], rows[column]
        pivot_value = rows[column][column]
        for entry in range(column, 4):
            rows[column][entry] /= pivot_value
        for row in range(3):
            if row == column:
                continue
            factor = rows[row][column]
            for entry in range(column, 4):
                rows[row][entry] -= factor * rows[column][entry]
    return rows[0][3], rows[1][3], rows[2][3]


def _eval_quadratic(coefficients: tuple[float, float, float], t: float) -> float:
    a, b, c = coefficients
    return a * t * t + b * t + c


def _eval_quadratic_derivative(coefficients: tuple[float, float, float], t: float) -> float:
    a, b, _ = coefficients
    return 2.0 * a * t + b


def _track_quality(detected: list[bool], confidences: list[float], index: int, radius: int = 2) -> float:
    start = max(0, index - radius)
    end = min(len(detected), index + radius + 1)
    if start >= end:
        return 0.0
    local_detected = sum(1 for value in detected[start:end] if value) / (end - start)
    local_confidence = sum(confidences[start:end]) / (end - start)
    return _clamp01(0.35 + 0.45 * local_detected + 0.20 * local_confidence)


def _missing_edge_score(detected: list[bool], index: int) -> float:
    if not detected[index]:
        return 0.0
    score = 0.0
    if index > 0 and not detected[index - 1]:
        score += 0.5
    if index < len(detected) - 1 and not detected[index + 1]:
        score += 0.5
    return score


def _robust_scale(values: list[float]) -> float:
    positives = sorted(value for value in values if value > 0.0)
    if not positives:
        return 1.0
    return max(median(positives), 1.0)


def _ratio(value: float, scale: float) -> float:
    if scale <= 0.0:
        return 0.0
    return _clamp01(value / scale)


def _clamp01(value: float) -> float:
    return _clamp(value, 0.0, 1.0)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
