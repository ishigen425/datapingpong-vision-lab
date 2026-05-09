from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class HeatmapPeak:
    x: int
    y: int
    score: float


@dataclass(frozen=True)
class TrackObservation:
    x: int | None
    y: int | None
    confidence: float
    candidate_count: int


class BallHeatmapTracker:
    def __init__(
        self,
        *,
        max_distance: float = 48.0,
        distance_weight: float = 0.015,
        process_noise: float = 2.0,
        measurement_noise: float = 6.0,
        max_missed_frames: int = 3,
    ) -> None:
        self.max_distance = max_distance
        self.distance_weight = distance_weight
        self.max_missed_frames = max_missed_frames
        self._state: np.ndarray | None = None
        self._covariance: np.ndarray | None = None
        self._process_noise = float(process_noise)
        self._measurement_noise = float(measurement_noise)
        self._missed_frames = 0

    def step(self, candidates: list[HeatmapPeak]) -> TrackObservation:
        if self._state is None:
            if not candidates:
                return TrackObservation(x=None, y=None, confidence=0.0, candidate_count=0)
            selected = max(candidates, key=lambda candidate: candidate.score)
            self._initialize(selected)
            return TrackObservation(
                x=selected.x,
                y=selected.y,
                confidence=selected.score,
                candidate_count=len(candidates),
            )

        predicted_state, predicted_covariance = self._predict()
        selected = self._select_candidate(predicted_state, candidates)
        if selected is None:
            self._state = predicted_state
            self._covariance = predicted_covariance
            self._missed_frames += 1
            if self._missed_frames > self.max_missed_frames:
                self._state = None
                self._covariance = None
            return TrackObservation(x=None, y=None, confidence=0.0, candidate_count=len(candidates))

        updated_state, updated_covariance = self._update(predicted_state, predicted_covariance, selected)
        self._state = updated_state
        self._covariance = updated_covariance
        self._missed_frames = 0
        return TrackObservation(
            x=selected.x,
            y=selected.y,
            confidence=selected.score,
            candidate_count=len(candidates),
        )

    def _initialize(self, selected: HeatmapPeak) -> None:
        self._state = np.asarray([float(selected.x), float(selected.y), 0.0, 0.0], dtype=np.float64)
        self._covariance = np.diag([16.0, 16.0, 64.0, 64.0]).astype(np.float64)
        self._missed_frames = 0

    def _predict(self) -> tuple[np.ndarray, np.ndarray]:
        assert self._state is not None
        assert self._covariance is not None
        transition = np.asarray(
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        process = np.diag(
            [
                self._process_noise**2,
                self._process_noise**2,
                (self._process_noise * 0.5) ** 2,
                (self._process_noise * 0.5) ** 2,
            ]
        ).astype(np.float64)
        return transition @ self._state, transition @ self._covariance @ transition.T + process

    def _select_candidate(self, predicted_state: np.ndarray, candidates: list[HeatmapPeak]) -> HeatmapPeak | None:
        if not candidates:
            return None
        predicted_x = float(predicted_state[0])
        predicted_y = float(predicted_state[1])
        distance_limit = self.max_distance * (1.0 + 0.75 * self._missed_frames)
        scored: list[tuple[float, HeatmapPeak]] = []
        for candidate in candidates:
            distance = float(np.hypot(candidate.x - predicted_x, candidate.y - predicted_y))
            if distance > distance_limit:
                continue
            scored.append((candidate.score - self.distance_weight * distance, candidate))
        if not scored:
            return None
        scored.sort(key=lambda row: row[0], reverse=True)
        return scored[0][1]

    def _update(
        self,
        predicted_state: np.ndarray,
        predicted_covariance: np.ndarray,
        selected: HeatmapPeak,
    ) -> tuple[np.ndarray, np.ndarray]:
        observation_matrix = np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        measurement_covariance = np.diag([self._measurement_noise**2, self._measurement_noise**2]).astype(np.float64)
        measurement = np.asarray([float(selected.x), float(selected.y)], dtype=np.float64)
        innovation = measurement - observation_matrix @ predicted_state
        innovation_covariance = observation_matrix @ predicted_covariance @ observation_matrix.T + measurement_covariance
        kalman_gain = predicted_covariance @ observation_matrix.T @ np.linalg.inv(innovation_covariance)
        identity = np.eye(4, dtype=np.float64)
        updated_state = predicted_state + kalman_gain @ innovation
        updated_covariance = (identity - kalman_gain @ observation_matrix) @ predicted_covariance
        return updated_state, updated_covariance


class BallBeliefHeatmapTracker:
    def __init__(
        self,
        shape: tuple[int, int],
        *,
        gravity_y: float = 0.18,
        measurement_floor: float = 0.02,
        measurement_power: float = 1.5,
        prior_blur_sigma: float = 3.0,
        max_missed_frames: int = 100,
        output_threshold: float = 0.15,
        centroid_radius: int = 5,
        reacquire_after_missed: bool = True,
        reacquire_threshold: float | None = None,
    ) -> None:
        self.shape = shape
        self.gravity_y = float(gravity_y)
        self.measurement_floor = float(measurement_floor)
        self.measurement_power = float(measurement_power)
        self.prior_blur_sigma = float(prior_blur_sigma)
        self.max_missed_frames = int(max_missed_frames)
        self.output_threshold = float(output_threshold)
        self.centroid_radius = int(centroid_radius)
        self.reacquire_after_missed = bool(reacquire_after_missed)
        self.reacquire_threshold = float(reacquire_threshold if reacquire_threshold is not None else output_threshold)
        self._belief: np.ndarray | None = None
        self._position: np.ndarray | None = None
        self._velocity = np.zeros(2, dtype=np.float64)
        self._acceleration = np.zeros(2, dtype=np.float64)
        self._missed_frames = 0

    def step(self, heatmap: np.ndarray, *, peaks: list[HeatmapPeak]) -> TrackObservation:
        measurement = self._prepare_measurement(heatmap)
        measurement_peak = float(np.max(measurement)) if measurement.size else 0.0
        reacquired_peak = self._select_reacquire_peak(peaks)
        if self._belief is not None and reacquired_peak is not None:
            posterior = self._normalize(measurement)
            self._belief = posterior
            self._reset_dynamics(np.asarray([float(reacquired_peak.x), float(reacquired_peak.y)], dtype=np.float64))
            self._missed_frames = 0
            return TrackObservation(
                x=reacquired_peak.x,
                y=reacquired_peak.y,
                confidence=reacquired_peak.score,
                candidate_count=len(peaks),
            )
        if self._belief is None:
            if measurement_peak <= 0.0:
                return TrackObservation(x=None, y=None, confidence=0.0, candidate_count=len(peaks))
            posterior = self._normalize(measurement)
        else:
            prior = self._predict_prior()
            posterior = self._normalize(prior * (measurement + self.measurement_floor))

        x_value, y_value = self._select_output_position(posterior, heatmap, peaks)
        confidence = float(heatmap[y_value, x_value]) if x_value is not None and y_value is not None else 0.0
        if x_value is None or y_value is None:
            self._belief = None
            self._position = None
            self._velocity[:] = 0.0
            self._acceleration[:] = 0.0
            return TrackObservation(x=None, y=None, confidence=0.0, candidate_count=len(peaks))

        posterior_peak = float(posterior[y_value, x_value])
        detected = confidence >= self.output_threshold
        self._belief = posterior
        self._update_dynamics(np.asarray([float(x_value), float(y_value)], dtype=np.float64), measurement_peak=measurement_peak)
        if not detected:
            self._missed_frames += 1
            if self._missed_frames > self.max_missed_frames:
                self._belief = None
                self._position = None
                self._velocity[:] = 0.0
                self._acceleration[:] = 0.0
            return TrackObservation(
                x=x_value,
                y=y_value,
                confidence=max(confidence * 0.5, posterior_peak),
                candidate_count=len(peaks),
            )
        self._missed_frames = 0
        return TrackObservation(x=x_value, y=y_value, confidence=confidence, candidate_count=len(peaks))

    def _select_reacquire_peak(self, peaks: list[HeatmapPeak]) -> HeatmapPeak | None:
        if not self.reacquire_after_missed or self._missed_frames <= 0 or not peaks:
            return None
        selected = max(peaks, key=lambda peak: peak.score)
        if selected.score < self.reacquire_threshold:
            return None
        return selected

    def _prepare_measurement(self, heatmap: np.ndarray) -> np.ndarray:
        measurement = np.asarray(heatmap, dtype=np.float32)
        measurement = np.clip(measurement, 0.0, 1.0)
        return np.power(measurement, self.measurement_power)

    def _predict_prior(self) -> np.ndarray:
        assert self._belief is not None
        shifted = shift_heatmap(
            self._belief,
            dx=float(self._velocity[0] + 0.5 * self._acceleration[0]),
            dy=float(self._velocity[1] + 0.5 * (self._acceleration[1] + self.gravity_y)),
        )
        sigma = self.prior_blur_sigma + 0.25 * float(np.hypot(self._velocity[0], self._velocity[1])) + 0.7 * min(self._missed_frames, 12)
        if sigma > 0.0:
            shifted = cv2.GaussianBlur(shifted, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_CONSTANT)
        if self._position is not None:
            predicted_center = (
                float(self._position[0] + self._velocity[0] + 0.5 * self._acceleration[0]),
                float(self._position[1] + self._velocity[1] + 0.5 * (self._acceleration[1] + self.gravity_y)),
            )
            center_prior = gaussian_heatmap(
                self.shape,
                center=predicted_center,
                sigma=max(2.0, sigma),
            )
            shifted = 0.7 * shifted + 0.3 * center_prior
        return self._normalize(shifted)

    def _estimate_position(self, posterior: np.ndarray) -> tuple[int | None, int | None]:
        if posterior.size == 0:
            return None, None
        flat_index = int(np.argmax(posterior))
        peak_value = float(posterior.flat[flat_index])
        if peak_value <= 0.0:
            return None, None
        peak_y, peak_x = np.unravel_index(flat_index, posterior.shape)
        x0 = max(0, peak_x - self.centroid_radius)
        x1 = min(posterior.shape[1], peak_x + self.centroid_radius + 1)
        y0 = max(0, peak_y - self.centroid_radius)
        y1 = min(posterior.shape[0], peak_y + self.centroid_radius + 1)
        window = posterior[y0:y1, x0:x1]
        if float(window.sum()) <= 0.0:
            return int(peak_x), int(peak_y)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        centroid_x = float((window * xx).sum() / window.sum())
        centroid_y = float((window * yy).sum() / window.sum())
        return int(round(centroid_x)), int(round(centroid_y))

    def _select_output_position(
        self,
        posterior: np.ndarray,
        raw_heatmap: np.ndarray,
        peaks: list[HeatmapPeak],
    ) -> tuple[int | None, int | None]:
        predicted_center = self._predicted_center()
        max_jump = self._max_jump_distance()
        if peaks:
            scored = []
            for peak in peaks:
                if predicted_center is not None:
                    distance = float(np.hypot(peak.x - predicted_center[0], peak.y - predicted_center[1]))
                    if distance > max_jump:
                        continue
                scored.append((float(posterior[peak.y, peak.x]) + 0.1 * float(raw_heatmap[peak.y, peak.x]), peak))
            scored.sort(key=lambda row: row[0], reverse=True)
            if scored:
                best_score, best_peak = scored[0]
                if best_score > 0.0:
                    return best_peak.x, best_peak.y
        estimate_x, estimate_y = self._estimate_position(posterior)
        if predicted_center is not None and estimate_x is not None and estimate_y is not None:
            distance = float(np.hypot(estimate_x - predicted_center[0], estimate_y - predicted_center[1]))
            if distance > max_jump:
                return self._clamp_point(int(round(predicted_center[0])), int(round(predicted_center[1])))
        return estimate_x, estimate_y

    def _predicted_center(self) -> tuple[float, float] | None:
        if self._position is None:
            return None
        return (
            float(self._position[0] + self._velocity[0] + 0.5 * self._acceleration[0]),
            float(self._position[1] + self._velocity[1] + 0.5 * (self._acceleration[1] + self.gravity_y)),
        )

    def _max_jump_distance(self) -> float:
        speed = float(np.hypot(self._velocity[0], self._velocity[1]))
        return min(48.0, 8.0 + 1.35 * speed + 4.0 * min(self._missed_frames, 6))

    def _clamp_point(self, x: int, y: int) -> tuple[int, int]:
        width = self.shape[1]
        height = self.shape[0]
        return max(0, min(width - 1, x)), max(0, min(height - 1, y))

    def _update_dynamics(self, position: np.ndarray, *, measurement_peak: float) -> None:
        if self._position is None:
            self._reset_dynamics(position)
            return
        observed_velocity = position - self._position
        predicted_velocity = self._velocity + np.asarray([0.0, self.gravity_y], dtype=np.float64)
        trust = 0.55 if measurement_peak >= self.output_threshold else 0.2
        new_velocity = (1.0 - trust) * predicted_velocity + trust * observed_velocity
        observed_acceleration = new_velocity - self._velocity
        self._acceleration = 0.75 * self._acceleration + 0.25 * observed_acceleration
        self._acceleration[1] = 0.6 * self._acceleration[1] + 0.4 * self.gravity_y
        self._velocity = new_velocity
        self._position = position

    def _reset_dynamics(self, position: np.ndarray) -> None:
        self._position = position
        self._velocity[:] = 0.0
        self._acceleration[:] = np.asarray([0.0, self.gravity_y], dtype=np.float64)

    @staticmethod
    def _normalize(heatmap: np.ndarray) -> np.ndarray:
        total = float(np.sum(heatmap))
        if total <= 0.0:
            return np.zeros_like(heatmap, dtype=np.float32)
        return np.asarray(heatmap / total, dtype=np.float32)


def gaussian_heatmap(shape: tuple[int, int], *, center: tuple[float, float], sigma: float) -> np.ndarray:
    height, width = shape
    center_x, center_y = center
    if sigma <= 0.0:
        sigma = 1.0
    x0 = max(0, int(round(center_x - 3 * sigma)))
    x1 = min(width, int(round(center_x + 3 * sigma)) + 1)
    y0 = max(0, int(round(center_y - 3 * sigma)))
    y1 = min(height, int(round(center_y + 3 * sigma)) + 1)
    heatmap = np.zeros((height, width), dtype=np.float32)
    if x0 >= x1 or y0 >= y1:
        return heatmap
    yy, xx = np.mgrid[y0:y1, x0:x1]
    local = np.exp(-0.5 * (((xx - center_x) / sigma) ** 2 + ((yy - center_y) / sigma) ** 2))
    heatmap[y0:y1, x0:x1] = local.astype(np.float32)
    total = float(heatmap.sum())
    return heatmap / total if total > 0.0 else heatmap


def shift_heatmap(heatmap: np.ndarray, *, dx: float, dy: float) -> np.ndarray:
    matrix = np.asarray([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    shifted = cv2.warpAffine(
        np.asarray(heatmap, dtype=np.float32),
        matrix,
        (heatmap.shape[1], heatmap.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    return np.asarray(shifted, dtype=np.float32)


def extract_top_peaks(
    heatmap: np.ndarray,
    *,
    threshold: float,
    top_k: int,
    suppression_radius: int,
) -> list[HeatmapPeak]:
    working = np.asarray(heatmap, dtype=np.float64).copy()
    peaks: list[HeatmapPeak] = []
    for _ in range(max(0, top_k)):
        flat_index = int(np.argmax(working))
        score = float(working.flat[flat_index])
        if score < threshold:
            break
        y, x = np.unravel_index(flat_index, working.shape)
        peaks.append(HeatmapPeak(x=int(x), y=int(y), score=score))
        x0 = max(0, int(x) - suppression_radius)
        x1 = min(working.shape[1], int(x) + suppression_radius + 1)
        y0 = max(0, int(y) - suppression_radius)
        y1 = min(working.shape[0], int(y) + suppression_radius + 1)
        working[y0:y1, x0:x1] = 0.0
    return peaks
