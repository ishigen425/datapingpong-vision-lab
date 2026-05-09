from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from datapingpong.events import detect_event_peaks, score_trajectory
from datapingpong.events.trajectory import EventPeak, EventProbabilities
from datapingpong.events.io import load_ball_point_groups, peaks_to_dicts, probabilities_to_dicts
from datapingpong.events.table import load_table_geometry


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect bounce/hit events from ball coordinates.")
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data/annotations/ball_tracking/DJI_0056_001_predictions.json",
        help="Ball coordinates as legacy JSON, detector predictions JSON, OpenTTGames JSON, or normalized JSONL.",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/detect_events_from_ball/events.json")
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--input-frame-offset", type=int, default=0, help="Integer frame shift applied to input ball coordinates before scoring.")
    parser.add_argument("--bounce-threshold", type=float, default=0.35)
    parser.add_argument("--hit-threshold", type=float, default=0.65)
    parser.add_argument("--nms-window", type=int, default=8)
    parser.add_argument("--max-gap", type=int, default=3)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--hit-smooth-window", type=int, default=None)
    parser.add_argument("--table-geometry", type=Path, default=None, help="Optional JSON with table corner annotations.")
    parser.add_argument("--table-margin", type=float, default=0.05, help="Allowed normalized table margin for bounce candidates.")
    parser.add_argument("--input-frame-width", type=float, default=None, help="Coordinate width for input ball predictions.")
    parser.add_argument("--input-frame-height", type=float, default=None, help="Coordinate height for input ball predictions.")
    parser.add_argument("--hit-suppression-window", type=int, default=0)
    parser.add_argument("--arbitration-window", type=int, default=6)
    parser.add_argument("--hit-local-window", type=int, default=6, help="Frame radius used for hit local motion quality checks.")
    parser.add_argument("--hit-min-local-x-span", type=float, default=0.0, help="Minimum local x span in input-coordinate pixels for hit candidates.")
    parser.add_argument("--hit-min-local-y-span", type=float, default=0.0, help="Minimum local y span in input-coordinate pixels for hit candidates.")
    parser.add_argument("--hit-min-local-detections", type=int, default=0, help="Minimum detected ball points in the local hit quality window.")
    parser.add_argument(
        "--hit-min-directional-x-displacement",
        type=float,
        default=0.0,
        help="Minimum pre/post x displacement in input-coordinate pixels, with opposite signs, for hit candidates.",
    )
    parser.add_argument(
        "--hit-reject-same-directional-x-displacement",
        type=float,
        default=0.0,
        help="Reject hit candidates whose pre/post x displacements have the same sign and both exceed this input-pixel threshold.",
    )
    args = parser.parse_args()

    groups = load_ball_point_groups(
        args.input,
        confidence_threshold=args.confidence_threshold,
        frame_offset=args.input_frame_offset,
    )
    table_geometry = load_table_geometry(args.table_geometry)
    probabilities = []
    peaks = []
    for item, points in groups.items():
        item_probabilities = score_trajectory(points, max_gap=args.max_gap, smooth_window=args.smooth_window)
        if args.hit_smooth_window is not None and args.hit_smooth_window != args.smooth_window:
            hit_probabilities = score_trajectory(points, max_gap=args.max_gap, smooth_window=args.hit_smooth_window)
            item_probabilities = merge_hit_probabilities(item_probabilities, hit_probabilities)
        input_width = None
        input_height = None
        if table_geometry is not None:
            input_width, input_height = resolve_input_frame_size(
                points,
                table_geometry,
                width=args.input_frame_width,
                height=args.input_frame_height,
            )
            item_probabilities = apply_table_event_prior(
                item_probabilities,
                table_geometry,
                input_width=input_width,
                input_height=input_height,
            )
        item_peaks = detect_event_peaks(
            item_probabilities,
            bounce_threshold=args.bounce_threshold,
            hit_threshold=args.hit_threshold,
            nms_window=args.nms_window,
        )
        item_peaks = filter_hits_by_local_motion(
            item_peaks,
            points,
            window=args.hit_local_window,
            min_x_span=args.hit_min_local_x_span,
            min_y_span=args.hit_min_local_y_span,
            min_detections=args.hit_min_local_detections,
            min_directional_x_displacement=args.hit_min_directional_x_displacement,
            reject_same_directional_x_displacement=args.hit_reject_same_directional_x_displacement,
        )
        if table_geometry is not None:
            item_peaks = [
                scale_peak(peak, input_width, input_height, table_geometry)
                for peak in item_peaks
            ]
            item_peaks = [
                peak
                for peak in item_peaks
                if peak.event != "bounce" or table_geometry.contains(peak.x, peak.y, margin=args.table_margin)
            ]
            item_peaks = arbitrate_table_events(item_peaks, table_geometry, window=args.arbitration_window)
        if args.hit_suppression_window > 0:
            item_peaks = suppress_bounces_near_hits(item_peaks, window=args.hit_suppression_window)
        for row in probabilities_to_dicts(item_probabilities):
            if item != "__default__":
                row["item"] = item
            probabilities.append(row)
        for row in peaks_to_dicts(item_peaks):
            if item != "__default__":
                row["item"] = item
            peaks.append(row)

    payload = {
        "input": str(args.input),
        "confidence_threshold": args.confidence_threshold,
        "input_frame_offset": args.input_frame_offset,
        "bounce_threshold": args.bounce_threshold,
        "hit_threshold": args.hit_threshold,
        "nms_window": args.nms_window,
        "max_gap": args.max_gap,
        "smooth_window": args.smooth_window,
        "hit_smooth_window": args.hit_smooth_window,
        "table_geometry": str(args.table_geometry) if args.table_geometry else None,
        "table_margin": args.table_margin,
        "input_frame_width": args.input_frame_width,
        "input_frame_height": args.input_frame_height,
        "hit_suppression_window": args.hit_suppression_window,
        "arbitration_window": args.arbitration_window,
        "hit_local_window": args.hit_local_window,
        "hit_min_local_x_span": args.hit_min_local_x_span,
        "hit_min_local_y_span": args.hit_min_local_y_span,
        "hit_min_local_detections": args.hit_min_local_detections,
        "hit_min_directional_x_displacement": args.hit_min_directional_x_displacement,
        "hit_reject_same_directional_x_displacement": args.hit_reject_same_directional_x_displacement,
        "frames": probabilities,
        "predicted_events": sorted(peaks, key=lambda row: (row.get("item", "__default__"), row["frame"], row["event"])),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    counts = {}
    for peak in peaks:
        counts[peak["event"]] = counts.get(peak["event"], 0) + 1
    print(json.dumps({"output": str(args.output), "groups": len(groups), "frames": len(probabilities), "events": counts}, indent=2))
    return 0


def resolve_input_frame_size(
    points: list[Any],
    table_geometry: Any,
    *,
    width: float | None,
    height: float | None,
) -> tuple[float, float]:
    if width is not None and height is not None:
        return float(width), float(height)
    max_x = max(float(point.x or 0.0) for point in points)
    max_y = max(float(point.y or 0.0) for point in points)
    image_width = table_geometry.image_width or max_x
    image_height = table_geometry.image_height or max_y
    if max_x <= image_width / 2.0 + 2.0 and max_y <= image_height / 2.0 + 2.0:
        return image_width / 3.0, image_height / 3.0
    return image_width, image_height


def merge_hit_probabilities(
    bounce_rows: list[EventProbabilities],
    hit_rows: list[EventProbabilities],
) -> list[EventProbabilities]:
    hit_by_frame = {row.frame: row for row in hit_rows}
    merged = []
    for row in bounce_rows:
        hit_row = hit_by_frame.get(row.frame)
        merged.append(
            EventProbabilities(
                frame=row.frame,
                x=row.x,
                y=row.y,
                detected=row.detected,
                confidence=row.confidence,
                bounce_probability=row.bounce_probability,
                hit_probability=hit_row.hit_probability if hit_row is not None else row.hit_probability,
                speed=row.speed,
                acceleration=row.acceleration,
                turn_angle_degrees=row.turn_angle_degrees,
            )
        )
    return merged


def scale_peak(peak: EventPeak, input_width: float, input_height: float, table_geometry: Any) -> EventPeak:
    if peak.x is None or peak.y is None or table_geometry.image_width is None or table_geometry.image_height is None:
        return peak
    return replace(
        peak,
        x=float(peak.x) * float(table_geometry.image_width) / input_width,
        y=float(peak.y) * float(table_geometry.image_height) / input_height,
    )


def apply_table_event_prior(
    rows: list[EventProbabilities],
    table_geometry: Any,
    *,
    input_width: float,
    input_height: float,
) -> list[EventProbabilities]:
    if table_geometry.image_width is None or table_geometry.image_height is None:
        return rows
    adjusted = []
    for row in rows:
        if row.x is None or row.y is None:
            adjusted.append(row)
            continue
        video_x = float(row.x) * float(table_geometry.image_width) / input_width
        video_y = float(row.y) * float(table_geometry.image_height) / input_height
        table_x, table_y = table_geometry.image_to_table(video_x, video_y)
        outside_distance = _outside_unit_square_distance(table_x, table_y)
        edge_margin = min(table_x, 1.0 - table_x, table_y, 1.0 - table_y)
        on_table = outside_distance == 0.0
        near_edge = edge_margin < 0.10
        hit_multiplier = 1.0
        bounce_multiplier = 1.0
        if on_table and not near_edge:
            hit_multiplier *= 0.62
            bounce_multiplier *= 1.08
        elif near_edge:
            hit_multiplier *= 1.14
        if outside_distance > 0.0:
            hit_multiplier *= min(1.55, 1.18 + outside_distance * 1.8)
            bounce_multiplier *= max(0.25, 1.0 - outside_distance * 3.0)
        adjusted.append(
            EventProbabilities(
                frame=row.frame,
                x=row.x,
                y=row.y,
                detected=row.detected,
                confidence=row.confidence,
                bounce_probability=round(min(1.0, row.bounce_probability * bounce_multiplier), 6),
                hit_probability=round(min(1.0, row.hit_probability * hit_multiplier), 6),
                speed=row.speed,
                acceleration=row.acceleration,
                turn_angle_degrees=row.turn_angle_degrees,
            )
        )
    return adjusted


def filter_hits_by_local_motion(
    peaks: list[EventPeak],
    points: list[Any],
    *,
    window: int,
    min_x_span: float,
    min_y_span: float,
    min_detections: int,
    min_directional_x_displacement: float,
    reject_same_directional_x_displacement: float = 0.0,
) -> list[EventPeak]:
    if (
        min_x_span <= 0.0
        and min_y_span <= 0.0
        and min_detections <= 0
        and min_directional_x_displacement <= 0.0
        and reject_same_directional_x_displacement <= 0.0
    ):
        return peaks
    if window < 0:
        return peaks

    by_frame = {int(point.frame): point for point in points}
    kept = []
    for peak in peaks:
        if peak.event != "hit":
            kept.append(peak)
            continue
        local_points = [
            by_frame[frame]
            for frame in range(peak.frame - window, peak.frame + window + 1)
            if frame in by_frame
        ]
        detected_points = [
            point
            for point in local_points
            if point.detected and point.x is not None and point.y is not None
        ]
        local_stats = hit_local_motion_stats(peak, detected_points)
        if len(detected_points) < min_detections:
            continue
        x_span = local_stats["local_x_span"]
        y_span = local_stats["local_y_span"]
        if x_span < min_x_span or y_span < min_y_span:
            continue
        before_dx = local_stats["local_before_x_displacement"]
        after_dx = local_stats["local_after_x_displacement"]
        if min_directional_x_displacement > 0.0:
            if before_dx is None or after_dx is None:
                continue
            if before_dx * after_dx >= 0.0:
                continue
            if min(abs(before_dx), abs(after_dx)) < min_directional_x_displacement:
                continue
        if reject_same_directional_x_displacement > 0.0 and before_dx is not None and after_dx is not None:
            same_direction = before_dx * after_dx > 0.0
            strong_same_direction = min(abs(before_dx), abs(after_dx)) >= reject_same_directional_x_displacement
            if same_direction and strong_same_direction:
                continue
        kept.append(
            replace(
                peak,
                local_x_span=round(x_span, 6),
                local_y_span=round(y_span, 6),
                local_detections=local_stats["local_detections"],
                local_before_x_displacement=round(before_dx, 6) if before_dx is not None else None,
                local_after_x_displacement=round(after_dx, 6) if after_dx is not None else None,
            )
        )
    return kept


def hit_local_motion_stats(peak: EventPeak, detected_points: list[Any]) -> dict[str, Any]:
    if not detected_points:
        return {
            "local_x_span": 0.0,
            "local_y_span": 0.0,
            "local_detections": 0,
            "local_before_x_displacement": None,
            "local_after_x_displacement": None,
        }

    x_values = [float(point.x) for point in detected_points]
    y_values = [float(point.y) for point in detected_points]
    before_points = [point for point in detected_points if int(point.frame) <= peak.frame]
    after_points = [point for point in detected_points if int(point.frame) >= peak.frame]
    before_dx = None
    after_dx = None
    if before_points and peak.x is not None:
        before_dx = float(peak.x) - float(before_points[0].x)
    if after_points and peak.x is not None:
        after_dx = float(after_points[-1].x) - float(peak.x)
    return {
        "local_x_span": max(x_values) - min(x_values),
        "local_y_span": max(y_values) - min(y_values),
        "local_detections": len(detected_points),
        "local_before_x_displacement": before_dx,
        "local_after_x_displacement": after_dx,
    }


def arbitrate_table_events(peaks: list[EventPeak], table_geometry: Any, *, window: int) -> list[EventPeak]:
    if window < 0:
        return peaks
    removed: set[int] = set()
    indexed = list(enumerate(peaks))
    for bounce_index, bounce in indexed:
        if bounce.event != "bounce" or bounce_index in removed:
            continue
        for hit_index, hit in indexed:
            if hit.event != "hit" or hit_index in removed or abs(bounce.frame - hit.frame) > window:
                continue
            bounce_zone = table_zone(bounce, table_geometry)
            hit_zone = table_zone(hit, table_geometry)
            if hit_zone != "inside" and bounce_zone != "inside":
                removed.add(bounce_index)
            elif bounce_zone == "inside" and hit_zone == "inside":
                if hit.probability > bounce.probability * 1.35:
                    removed.add(bounce_index)
                else:
                    removed.add(hit_index)
            elif bounce_zone == "inside":
                removed.add(hit_index)
            else:
                removed.add(bounce_index)
    return [peak for index, peak in indexed if index not in removed]


def table_zone(peak: EventPeak, table_geometry: Any) -> str:
    if peak.x is None or peak.y is None:
        return "unknown"
    table_x, table_y = table_geometry.image_to_table(float(peak.x), float(peak.y))
    outside_distance = _outside_unit_square_distance(table_x, table_y)
    if outside_distance > 0.0:
        return "outside"
    edge_margin = min(table_x, 1.0 - table_x, table_y, 1.0 - table_y)
    return "edge" if edge_margin < 0.10 else "inside"


def _outside_unit_square_distance(x: float, y: float) -> float:
    dx = max(0.0 - x, 0.0, x - 1.0)
    dy = max(0.0 - y, 0.0, y - 1.0)
    return (dx * dx + dy * dy) ** 0.5


def suppress_bounces_near_hits(peaks: list[EventPeak], *, window: int) -> list[EventPeak]:
    hit_frames = [peak.frame for peak in peaks if peak.event == "hit"]
    if not hit_frames or window < 0:
        return peaks
    return [
        peak
        for peak in peaks
        if peak.event != "bounce" or all(abs(peak.frame - hit_frame) > window for hit_frame in hit_frames)
    ]


if __name__ == "__main__":
    raise SystemExit(main())
