from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .rally import RallySegment


@dataclass(frozen=True)
class RallyProposalFeatureTable:
    rows: list[dict[str, Any]]
    feature_names: list[str]
    features: np.ndarray


def build_rally_proposal_feature_table(proposals: list[RallySegment]) -> RallyProposalFeatureTable:
    feature_names = [
        "duration_frames",
        "detected_points",
        "detected_density",
        "event_count",
        "bounce_count",
        "hit_count",
        "has_events",
        "first_event_offset",
        "last_event_offset",
        "span_x",
        "span_y",
        "serve_like_score",
        "serve_like_start",
        "pose_evidence_frames",
        "toss_like_start",
        "toss_rise_px",
        "toss_x_span_px",
        "bounce_x_direction_changes",
        "max_bounce_x_delta",
        "start_nearest_wrist_distance",
        "end_nearest_wrist_distance",
    ]
    rows: list[dict[str, Any]] = []
    feature_rows: list[list[float]] = []
    for index, proposal in enumerate(proposals, start=1):
        event_counts = _event_counts(proposal)
        first_event_frame = min((int(event["frame"]) for event in proposal.events), default=None)
        last_event_frame = max((int(event["frame"]) for event in proposal.events), default=None)
        span_x = _span(proposal.min_x, proposal.max_x)
        span_y = _span(proposal.min_y, proposal.max_y)
        duration_frames = proposal.duration_frames
        row = {
            "proposal_id": index,
            "start_frame": proposal.start_frame,
            "end_frame": proposal.end_frame,
            "track_start_frame": proposal.track_start_frame,
            "track_end_frame": proposal.track_end_frame,
            "duration_frames": duration_frames,
            "detected_points": proposal.detected_points,
            "event_count": len(proposal.events),
            "bounce_count": event_counts["bounce"],
            "hit_count": event_counts["hit"],
            "first_event_frame": first_event_frame,
            "last_event_frame": last_event_frame,
            "first_event_offset": None if first_event_frame is None else first_event_frame - proposal.track_start_frame,
            "last_event_offset": None if last_event_frame is None else last_event_frame - proposal.track_start_frame,
            "span_x": span_x,
            "span_y": span_y,
            "serve_like_start": proposal.serve_like_start,
            "serve_like_score": proposal.serve_like_score,
            "pose_evidence_frames": proposal.pose_evidence_frames,
            "toss_like_start": proposal.toss_like_start,
            "toss_rise_px": proposal.toss_rise_px,
            "toss_x_span_px": proposal.toss_x_span_px,
            "bounce_x_direction_changes": proposal.bounce_x_direction_changes,
            "max_bounce_x_delta": proposal.max_bounce_x_delta,
            "start_nearest_wrist_distance": proposal.start_nearest_wrist_distance,
            "end_nearest_wrist_distance": proposal.end_nearest_wrist_distance,
            "rule_keep": proposal.rule_keep,
            "rule_reject_reason": proposal.rule_reject_reason,
            "label": None,
        }
        rows.append(row)
        feature_rows.append(
            [
                float(duration_frames),
                float(proposal.detected_points),
                float(proposal.detected_points) / max(1.0, float(duration_frames)),
                float(len(proposal.events)),
                float(event_counts["bounce"]),
                float(event_counts["hit"]),
                1.0 if proposal.events else 0.0,
                float(row["first_event_offset"] or 0.0),
                float(row["last_event_offset"] or 0.0),
                float(span_x),
                float(span_y),
                float(proposal.serve_like_score or 0.0),
                1.0 if proposal.serve_like_start else 0.0,
                float(proposal.pose_evidence_frames),
                1.0 if proposal.toss_like_start else 0.0,
                float(proposal.toss_rise_px or 0.0),
                float(proposal.toss_x_span_px or 0.0),
                float(proposal.bounce_x_direction_changes),
                float(proposal.max_bounce_x_delta or 0.0),
                float(proposal.start_nearest_wrist_distance or 0.0),
                float(proposal.end_nearest_wrist_distance or 0.0),
            ]
        )
    return RallyProposalFeatureTable(rows=rows, feature_names=feature_names, features=np.asarray(feature_rows, dtype=np.float64))


def labeled_rows_to_matrix(
    rows: list[dict[str, Any]],
    feature_names: list[str],
    *,
    label_field: str = "label",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    labeled = [row for row in rows if row.get(label_field) not in {None, "", "unlabeled"}]
    if not labeled:
        raise ValueError("No labeled proposal rows found.")
    classes = sorted({str(row[label_field]) for row in labeled})
    class_to_index = {name: index for index, name in enumerate(classes)}
    x = np.asarray([[float(row.get(name, 0.0) or 0.0) for name in feature_names] for row in labeled], dtype=np.float64)
    y = np.asarray([class_to_index[str(row[label_field])] for row in labeled], dtype=np.int64)
    return x, y, classes


def _event_counts(proposal: RallySegment) -> dict[str, int]:
    counts = {"bounce": 0, "hit": 0}
    for event in proposal.events:
        name = str(event["event"])
        if name in counts:
            counts[name] += 1
    return counts


def _span(min_value: float | None, max_value: float | None) -> float:
    if min_value is None or max_value is None:
        return 0.0
    return float(max_value - min_value)
