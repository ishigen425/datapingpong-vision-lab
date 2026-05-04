"""Event detection from ball trajectories."""

from .trajectory import (
    BallPoint,
    EventPeak,
    EventProbabilities,
    detect_event_peaks,
    score_trajectory,
)

__all__ = [
    "BallPoint",
    "EventPeak",
    "EventProbabilities",
    "detect_event_peaks",
    "score_trajectory",
]
