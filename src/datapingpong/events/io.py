from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .trajectory import BallPoint, EventPeak, EventProbabilities


def load_ball_points(path: Path, *, confidence_threshold: float = 0.5) -> list[BallPoint]:
    groups = load_ball_point_groups(path, confidence_threshold=confidence_threshold)
    if len(groups) == 1:
        return next(iter(groups.values()))
    points: list[BallPoint] = []
    for group_points in groups.values():
        points.extend(group_points)
    return sorted(points, key=lambda point: point.frame)


def load_ball_point_groups(path: Path, *, confidence_threshold: float = 0.5) -> dict[str, list[BallPoint]]:
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        groups: dict[str, list[BallPoint]] = {}
        for index, row in enumerate(rows):
            item = str(row.get("item", "__default__"))
            groups.setdefault(item, []).append(_row_to_ball_point(row, index, confidence_threshold))
        return {item: sorted(points, key=lambda point: point.frame) for item, points in groups.items()}

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "predictions" in payload:
        return {"__default__": [_row_to_ball_point(row, index, confidence_threshold) for index, row in enumerate(payload["predictions"])]}
    if isinstance(payload, list):
        return {"__default__": [_row_to_ball_point(row, index, confidence_threshold) for index, row in enumerate(payload)]}
    if isinstance(payload, dict):
        return {"__default__": [_mapping_item_to_ball_point(frame, row, confidence_threshold) for frame, row in payload.items()]}
    raise ValueError(f"Unsupported ball input format: {path}")


def load_reference_events(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        rows = json.loads(path.read_text(encoding="utf-8"))

    events: list[dict[str, Any]] = []
    if isinstance(rows, dict):
        for key, value in rows.items():
            if key.endswith("_frames") and isinstance(value, list):
                event = key[: -len("_frames")]
                events.extend({"event": event, "frame": int(frame), "item": "__default__"} for frame in value)
            elif str(key).isdigit() and isinstance(value, str):
                events.append({"event": _normalize_event(value), "frame": int(key), "item": "__default__"})
        return sorted(events, key=lambda row: (row["frame"], row["event"]))

    if isinstance(rows, list):
        for row in rows:
            event = _normalize_event(str(row["event"]))
            if event in {"bounce", "hit", "net_hit", "empty"}:
                events.append({"event": event, "frame": int(row["frame"]), "item": str(row.get("item", "__default__"))})
        return sorted(events, key=lambda row: (row["frame"], row["event"]))
    raise ValueError(f"Unsupported event input format: {path}")


def probabilities_to_dicts(rows: list[EventProbabilities]) -> list[dict[str, Any]]:
    return [
        {
            "frame": row.frame,
            "x": row.x,
            "y": row.y,
            "detected": row.detected,
            "confidence": row.confidence,
            "bounce_probability": row.bounce_probability,
            "hit_probability": row.hit_probability,
            "speed": row.speed,
            "acceleration": row.acceleration,
            "turn_angle_degrees": row.turn_angle_degrees,
        }
        for row in rows
    ]


def peaks_to_dicts(peaks: list[EventPeak]) -> list[dict[str, Any]]:
    return [
        {
            "event": peak.event,
            "frame": peak.frame,
            "probability": peak.probability,
            "x": peak.x,
            "y": peak.y,
        }
        for peak in peaks
    ]


def _row_to_ball_point(row: dict[str, Any], index: int, confidence_threshold: float) -> BallPoint:
    frame = int(row.get("frame", index))
    confidence = float(row.get("confidence", row.get("prod", 1.0)))
    x = row.get("x")
    y = row.get("y")
    detected = bool(row.get("detected", True))
    if x is None or y is None:
        detected = False
    else:
        x = float(x)
        y = float(y)
        if (x == 0.0 and y == 0.0) or (x == -1.0 and y == -1.0):
            detected = False
    detected = detected and confidence >= confidence_threshold
    return BallPoint(frame=frame, x=x if detected else None, y=y if detected else None, confidence=confidence, detected=detected)


def _mapping_item_to_ball_point(frame: str, row: Any, confidence_threshold: float) -> BallPoint:
    if not isinstance(row, dict):
        raise ValueError(f"Expected coordinate mapping for frame {frame}, got {type(row).__name__}")
    payload = {"frame": int(frame), **row}
    return _row_to_ball_point(payload, int(frame), confidence_threshold)


def _normalize_event(event: str) -> str:
    normalized = event.strip().lower().replace(" ", "_").replace("-", "_")
    if normalized in {"net", "net_hit", "net_hit_event"}:
        return "net_hit"
    if normalized == "empty_event":
        return "empty"
    return normalized
