"""Event detection from ball trajectories."""

from .trajectory import (
    BallPoint,
    EventPeak,
    EventProbabilities,
    detect_event_peaks,
    score_trajectory,
)
from .rally import RallySegment, build_rally_proposals, detect_rallies, rallies_to_dicts
from .stroke import HitStroke, infer_hit_strokes, strokes_to_dicts
from .table import TableGeometry, load_table_geometry, table_geometry_from_dict

__all__ = [
    "BallPoint",
    "EventPeak",
    "EventProbabilities",
    "HitStroke",
    "RallySegment",
    "TableGeometry",
    "build_rally_proposals",
    "detect_event_peaks",
    "detect_rallies",
    "infer_hit_strokes",
    "load_table_geometry",
    "rallies_to_dicts",
    "score_trajectory",
    "strokes_to_dicts",
    "table_geometry_from_dict",
]
