"""Event detection from ball trajectories."""

from .trajectory import (
    BallPoint,
    EventPeak,
    EventProbabilities,
    detect_event_peaks,
    score_trajectory,
)
from .table import TableGeometry, load_table_geometry, table_geometry_from_dict

__all__ = [
    "BallPoint",
    "EventPeak",
    "EventProbabilities",
    "TableGeometry",
    "detect_event_peaks",
    "load_table_geometry",
    "score_trajectory",
    "table_geometry_from_dict",
]
