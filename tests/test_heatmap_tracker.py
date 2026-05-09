from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datapingpong.tracking import BallBeliefHeatmapTracker, BallHeatmapTracker, extract_top_peaks


def test_extract_top_peaks_returns_separated_candidates() -> None:
    heatmap = np.zeros((12, 12), dtype=np.float32)
    heatmap[2, 3] = 0.9
    heatmap[8, 9] = 0.7
    heatmap[2, 4] = 0.8

    peaks = extract_top_peaks(heatmap, threshold=0.2, top_k=3, suppression_radius=1)

    assert [(peak.x, peak.y) for peak in peaks] == [(3, 2), (9, 8)]


def test_ball_heatmap_tracker_prefers_smooth_path() -> None:
    tracker = BallHeatmapTracker(max_distance=20.0, distance_weight=0.03, max_missed_frames=2)
    sequence = [
        [(10, 20, 0.55)],
        [(12, 20, 0.56)],
        [(14, 20, 0.52), (80, 80, 0.9)],
        [(16, 20, 0.53)],
    ]

    outputs = []
    for frame in sequence:
        peaks = [type("Peak", (), {"x": x, "y": y, "score": score})() for x, y, score in frame]
        observation = tracker.step(peaks)  # type: ignore[arg-type]
        outputs.append((observation.x, observation.y))

    assert outputs == [(10, 20), (12, 20), (14, 20), (16, 20)]


def test_belief_tracker_resists_far_spike() -> None:
    tracker = BallBeliefHeatmapTracker((48, 48), gravity_y=0.0, output_threshold=0.1)
    first = np.zeros((48, 48), dtype=np.float32)
    first[20, 10] = 0.9
    second = np.zeros((48, 48), dtype=np.float32)
    second[20, 12] = 0.6
    second[38, 40] = 0.95

    peaks1 = extract_top_peaks(first, threshold=0.1, top_k=3, suppression_radius=1)
    peaks2 = extract_top_peaks(second, threshold=0.1, top_k=3, suppression_radius=1)
    out1 = tracker.step(first, peaks=peaks1)
    out2 = tracker.step(second, peaks=peaks2)

    assert (out1.x, out1.y) == (10, 20)
    assert out2.x is not None
    assert out2.y is not None
    assert abs(out2.x - 12) <= 1
    assert abs(out2.y - 20) <= 1


def test_belief_tracker_keeps_local_state_on_weak_measurement() -> None:
    tracker = BallBeliefHeatmapTracker((48, 48), gravity_y=0.0, output_threshold=0.2)
    first = np.zeros((48, 48), dtype=np.float32)
    first[20, 10] = 0.9
    second = np.zeros((48, 48), dtype=np.float32)
    second[20, 11] = 0.18

    tracker.step(first, peaks=extract_top_peaks(first, threshold=0.1, top_k=3, suppression_radius=1))
    out2 = tracker.step(second, peaks=extract_top_peaks(second, threshold=0.1, top_k=3, suppression_radius=1))

    assert out2.x is not None
    assert out2.y is not None
    assert abs(out2.x - 11) <= 1
    assert abs(out2.y - 20) <= 1


def test_belief_tracker_reacquires_strong_peak_after_miss() -> None:
    tracker = BallBeliefHeatmapTracker((80, 80), gravity_y=0.0, output_threshold=0.2)
    first = np.zeros((80, 80), dtype=np.float32)
    first[20, 10] = 0.9
    missed = np.zeros((80, 80), dtype=np.float32)
    reacquired = np.zeros((80, 80), dtype=np.float32)
    reacquired[55, 60] = 0.85

    tracker.step(first, peaks=extract_top_peaks(first, threshold=0.1, top_k=3, suppression_radius=1))
    tracker.step(missed, peaks=[])
    out = tracker.step(reacquired, peaks=extract_top_peaks(reacquired, threshold=0.1, top_k=3, suppression_radius=1))

    assert (out.x, out.y) == (60, 55)
