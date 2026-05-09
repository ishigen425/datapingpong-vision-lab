from __future__ import annotations

import argparse
import csv
from html import escape
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from datapingpong.events.io import load_ball_point_groups, load_reference_events
from datapingpong.events.table import load_table_geometry


COLORS = {
    "x": "#2563eb",
    "y": "#dc2626",
    "bounce": "#16a34a",
    "hit": "#9333ea",
    "net_hit": "#0891b2",
    "empty": "#64748b",
    "grid": "#e2e8f0",
    "axis": "#334155",
    "missing": "#f97316",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot ball x/y coordinates and event markers as SVG.")
    parser.add_argument("--ball", type=Path, default=ROOT / "data/annotations/ball_tracking/DJI_0056_001_predictions.json")
    parser.add_argument("--events", type=Path, default=None, help="Optional predicted events JSON.")
    parser.add_argument("--reference", type=Path, default=None, help="Optional reference events JSON.")
    parser.add_argument("--table-geometry", type=Path, default=None, help="Optional table corner annotation JSON.")
    parser.add_argument("--input-frame-width", type=float, default=None, help="Coordinate width for input ball predictions.")
    parser.add_argument("--input-frame-height", type=float, default=None, help="Coordinate height for input ball predictions.")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/plot_ball_timeseries/DJI_0056_001_timeseries.svg")
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--ball-frame-offset", type=int, default=0)
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--item", type=str, default=None)
    parser.add_argument("--width", type=int, default=1800)
    parser.add_argument("--height", type=int, default=760)
    parser.add_argument("--diagnostics", type=Path, default=None, help="Optional details.csv from diagnose_event_detection.")
    parser.add_argument("--diagnostic-status", nargs="+", default=["false_positive", "false_negative"])
    parser.add_argument("--diagnostic-events", nargs="+", default=["bounce", "hit"])
    parser.add_argument("--window-radius", type=int, default=45)
    parser.add_argument("--max-windows", type=int, default=80)
    parser.add_argument("--windows-output-dir", type=Path, default=ROOT / "outputs/plot_ball_timeseries/windows")
    args = parser.parse_args()

    groups = load_ball_point_groups(
        args.ball,
        confidence_threshold=args.confidence_threshold,
        frame_offset=args.ball_frame_offset,
    )
    predicted_events = load_events(args.events) if args.events else []
    reference_events = load_reference_events(args.reference) if args.reference else []
    table_geometry = load_table_geometry(args.table_geometry)
    args.table_geometry_obj = table_geometry

    selected_items = [args.item] if args.item else sorted(groups)
    if args.diagnostics:
        written = write_diagnostic_windows(
            args,
            groups,
            predicted_events,
            reference_events,
            selected_items=selected_items,
        )
        print(json.dumps({"windows": written, "output_dir": str(args.windows_output_dir)}, indent=2))
        return 0

    if len(selected_items) != 1:
        raise ValueError("--item is required when the ball input contains multiple items")
    item = selected_items[0]
    svg = render_timeseries_svg(
        points=groups[item],
        predicted_events=events_for_item(predicted_events, item),
        reference_events=events_for_item(reference_events, item),
        table_geometry=table_geometry,
        input_frame_width=args.input_frame_width,
        input_frame_height=args.input_frame_height,
        title=f"Ball x/y timeseries: {item}",
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        width=args.width,
        height=args.height,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg, encoding="utf-8")
    print(json.dumps({"output": str(args.output), "item": item}, indent=2))
    return 0


def load_events(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("predicted_events"), list):
        rows = payload["predicted_events"]
    elif isinstance(payload, dict) and isinstance(payload.get("events"), list):
        rows = payload["events"]
    elif isinstance(payload, list):
        rows = payload
    else:
        raise ValueError(f"Unsupported event file: {path}")
    return [
        {
            **row,
            "event": str(row["event"]),
            "frame": int(row["frame"]),
            "item": str(row.get("item", "__default__")),
        }
        for row in rows
    ]


def write_diagnostic_windows(
    args: argparse.Namespace,
    groups: dict[str, list[Any]],
    predicted_events: list[dict[str, Any]],
    reference_events: list[dict[str, Any]],
    *,
    selected_items: list[str],
) -> int:
    rows = load_diagnostic_rows(
        args.diagnostics,
        statuses=set(args.diagnostic_status),
        events=set(args.diagnostic_events),
        selected_items=set(selected_items),
    )
    args.windows_output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for index, row in enumerate(rows[: args.max_windows], start=1):
        item = row["item"]
        frame = int(row["frame"])
        start = frame - args.window_radius
        end = frame + args.window_radius
        title = f"{row['status']} {row['event']} frame {frame} item {item}"
        svg = render_timeseries_svg(
            points=groups[item],
            predicted_events=events_for_item(predicted_events, item),
            reference_events=events_for_item(reference_events, item),
            table_geometry=args.table_geometry_obj,
            input_frame_width=args.input_frame_width,
            input_frame_height=args.input_frame_height,
            title=title,
            start_frame=start,
            end_frame=end,
            width=args.width,
            height=args.height,
            focus_frame=frame,
        )
        filename = f"{index:03d}_{safe_name(row['status'])}_{safe_name(row['event'])}_{frame}.svg"
        (args.windows_output_dir / filename).write_text(svg, encoding="utf-8")
        written += 1
    return written


def load_diagnostic_rows(
    path: Path,
    *,
    statuses: set[str],
    events: set[str],
    selected_items: set[str],
) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        row
        for row in rows
        if row.get("status") in statuses
        and row.get("event") in events
        and row.get("item", "__default__") in selected_items
        and row.get("frame")
    ]


def events_for_item(rows: list[dict[str, Any]], item: str) -> list[dict[str, Any]]:
    return [row for row in rows if str(row.get("item", "__default__")) == item]


def render_timeseries_svg(
    *,
    points: list[Any],
    predicted_events: list[dict[str, Any]],
    reference_events: list[dict[str, Any]],
    title: str,
    start_frame: int | None,
    end_frame: int | None,
    width: int,
    height: int,
    focus_frame: int | None = None,
    table_geometry: Any | None = None,
    input_frame_width: float | None = None,
    input_frame_height: float | None = None,
) -> str:
    visible_points = filter_points(points, start_frame=start_frame, end_frame=end_frame)
    if not visible_points:
        raise ValueError("No ball points in the requested frame range")

    min_frame = start_frame if start_frame is not None else min(int(point.frame) for point in visible_points)
    max_frame = end_frame if end_frame is not None else max(int(point.frame) for point in visible_points)
    if max_frame <= min_frame:
        max_frame = min_frame + 1

    values = [
        float(value)
        for point in visible_points
        for value in (point.x, point.y)
        if point.detected and value is not None
    ]
    y_min = min(values) if values else 0.0
    y_max = max(values) if values else 1.0
    if y_max <= y_min:
        y_max = y_min + 1.0
    margin = max(1.0, (y_max - y_min) * 0.08)
    y_min -= margin
    y_max += margin

    has_table_plot = table_geometry is not None
    plot = PlotArea(left=72, top=62, right=width - 28, bottom=height - (260 if has_table_plot else 76))
    table_plot = PlotArea(left=72, top=height - 212, right=width - 28, bottom=height - 76) if has_table_plot else None
    lines: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{plot.left}" y="30" font-family="sans-serif" font-size="18" fill="#0f172a">{escape(title)}</text>',
        f'<text x="{plot.left}" y="{height - 24}" font-family="sans-serif" font-size="12" fill="#475569">frame {min_frame} to {max_frame}</text>',
    ]
    lines.append(f'<text x="{plot.left}" y="{plot.top - 14}" font-family="sans-serif" font-size="13" fill="#334155">image coordinates</text>')
    lines.extend(grid_svg(plot, min_frame=min_frame, max_frame=max_frame, y_min=y_min, y_max=y_max))
    lines.extend(event_markers_svg(predicted_events, plot, min_frame, max_frame, source="predicted"))
    lines.extend(event_markers_svg(reference_events, plot, min_frame, max_frame, source="reference"))
    if focus_frame is not None:
        x = scale_x(focus_frame, plot, min_frame, max_frame)
        lines.append(f'<line x1="{x:.2f}" y1="{plot.top}" x2="{x:.2f}" y2="{plot.bottom}" stroke="#111827" stroke-width="2"/>')
    lines.extend(polyline_segments_svg(visible_points, "x", COLORS["x"], plot, min_frame, max_frame, y_min, y_max))
    lines.extend(polyline_segments_svg(visible_points, "y", COLORS["y"], plot, min_frame, max_frame, y_min, y_max))
    lines.extend(missing_markers_svg(visible_points, plot, min_frame, max_frame))
    if table_geometry is not None and table_plot is not None:
        table_points = table_relative_points(
            visible_points,
            table_geometry,
            input_frame_width=input_frame_width,
            input_frame_height=input_frame_height,
        )
        lines.append(f'<text x="{table_plot.left}" y="{table_plot.top - 14}" font-family="sans-serif" font-size="13" fill="#334155">table-relative coordinates</text>')
        lines.extend(grid_svg(table_plot, min_frame=min_frame, max_frame=max_frame, y_min=-0.25, y_max=1.25))
        lines.extend(table_bounds_svg(table_plot))
        lines.extend(event_markers_svg(predicted_events, table_plot, min_frame, max_frame, source="predicted"))
        lines.extend(event_markers_svg(reference_events, table_plot, min_frame, max_frame, source="reference"))
        if focus_frame is not None:
            x = scale_x(focus_frame, table_plot, min_frame, max_frame)
            lines.append(f'<line x1="{x:.2f}" y1="{table_plot.top}" x2="{x:.2f}" y2="{table_plot.bottom}" stroke="#111827" stroke-width="2"/>')
        lines.extend(table_polyline_segments_svg(table_points, "table_x", "#0ea5e9", table_plot, min_frame, max_frame))
        lines.extend(table_polyline_segments_svg(table_points, "table_y", "#f59e0b", table_plot, min_frame, max_frame))
        lines.extend(table_inside_markers_svg(table_points, table_plot, min_frame, max_frame))
    lines.extend(legend_svg(plot, include_table=has_table_plot))
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


class PlotArea:
    def __init__(self, *, left: int, top: int, right: int, bottom: int) -> None:
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


def filter_points(points: list[Any], *, start_frame: int | None, end_frame: int | None) -> list[Any]:
    return [
        point
        for point in points
        if (start_frame is None or int(point.frame) >= start_frame)
        and (end_frame is None or int(point.frame) <= end_frame)
    ]


def grid_svg(plot: PlotArea, *, min_frame: int, max_frame: int, y_min: float, y_max: float) -> list[str]:
    lines = [
        f'<rect x="{plot.left}" y="{plot.top}" width="{plot.width}" height="{plot.height}" fill="#f8fafc" stroke="{COLORS["axis"]}" stroke-width="1"/>'
    ]
    for index in range(6):
        frame = min_frame + (max_frame - min_frame) * index / 5
        x = scale_x(frame, plot, min_frame, max_frame)
        lines.append(f'<line x1="{x:.2f}" y1="{plot.top}" x2="{x:.2f}" y2="{plot.bottom}" stroke="{COLORS["grid"]}" stroke-width="1"/>')
        lines.append(f'<text x="{x:.2f}" y="{plot.bottom + 18}" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#475569">{frame:.0f}</text>')
    for index in range(6):
        value = y_min + (y_max - y_min) * index / 5
        y = scale_y(value, plot, y_min, y_max)
        lines.append(f'<line x1="{plot.left}" y1="{y:.2f}" x2="{plot.right}" y2="{y:.2f}" stroke="{COLORS["grid"]}" stroke-width="1"/>')
        lines.append(f'<text x="{plot.left - 8}" y="{y + 4:.2f}" text-anchor="end" font-family="sans-serif" font-size="10" fill="#475569">{value:.0f}</text>')
    return lines


def event_markers_svg(rows: list[dict[str, Any]], plot: PlotArea, min_frame: int, max_frame: int, *, source: str) -> list[str]:
    lines = []
    dash = "4 4" if source == "reference" else ""
    opacity = "0.85" if source == "predicted" else "0.55"
    for row in rows:
        frame = int(row["frame"])
        if frame < min_frame or frame > max_frame:
            continue
        event = str(row["event"])
        color = COLORS.get(event, "#0f172a")
        x = scale_x(frame, plot, min_frame, max_frame)
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        lines.append(f'<line x1="{x:.2f}" y1="{plot.top}" x2="{x:.2f}" y2="{plot.bottom}" stroke="{color}" stroke-width="1.5" opacity="{opacity}"{dash_attr}/>')
        label = f"{event[0].upper()}{'R' if source == 'reference' else 'P'}"
        lines.append(f'<text x="{x + 3:.2f}" y="{plot.top + 14}" font-family="sans-serif" font-size="10" fill="{color}" opacity="{opacity}">{escape(label)}</text>')
    return lines


def polyline_segments_svg(
    points: list[Any],
    attr: str,
    color: str,
    plot: PlotArea,
    min_frame: int,
    max_frame: int,
    y_min: float,
    y_max: float,
) -> list[str]:
    lines = []
    for segment in polyline_segments(points, attr):
        coords = [
            f"{scale_x(int(point.frame), plot, min_frame, max_frame):.2f},{scale_y(float(getattr(point, attr)), plot, y_min, y_max):.2f}"
            for point in segment
        ]
        if len(coords) == 1:
            cx, cy = coords[0].split(",", maxsplit=1)
            lines.append(f'<circle cx="{cx}" cy="{cy}" r="2" fill="{color}"/>')
        else:
            lines.append(f'<polyline points="{" ".join(coords)}" fill="none" stroke="{color}" stroke-width="2"/>')
    return lines


def polyline_segments(points: list[Any], attr: str) -> list[list[Any]]:
    segments: list[list[Any]] = []
    current: list[Any] = []
    for point in sorted(points, key=lambda row: int(row.frame)):
        value = getattr(point, attr)
        if point.detected and value is not None:
            current.append(point)
            continue
        if current:
            segments.append(current)
            current = []
    if current:
        segments.append(current)
    return segments


def missing_markers_svg(points: list[Any], plot: PlotArea, min_frame: int, max_frame: int) -> list[str]:
    lines = []
    for point in points:
        if point.detected:
            continue
        x = scale_x(int(point.frame), plot, min_frame, max_frame)
        lines.append(f'<line x1="{x:.2f}" y1="{plot.bottom - 14}" x2="{x:.2f}" y2="{plot.bottom}" stroke="{COLORS["missing"]}" stroke-width="1" opacity="0.6"/>')
    return lines


def table_relative_points(
    points: list[Any],
    table_geometry: Any,
    *,
    input_frame_width: float | None,
    input_frame_height: float | None,
) -> list[dict[str, Any]]:
    width, height = resolve_input_frame_size(
        points,
        table_geometry,
        width=input_frame_width,
        height=input_frame_height,
    )
    rows = []
    for point in points:
        row = {"frame": int(point.frame), "detected": bool(point.detected), "table_x": None, "table_y": None, "inside": False}
        if point.detected and point.x is not None and point.y is not None:
            video_x = float(point.x)
            video_y = float(point.y)
            if table_geometry.image_width is not None:
                video_x = video_x * float(table_geometry.image_width) / width
            if table_geometry.image_height is not None:
                video_y = video_y * float(table_geometry.image_height) / height
            table_x, table_y = table_geometry.image_to_table(video_x, video_y)
            row["table_x"] = table_x
            row["table_y"] = table_y
            row["inside"] = 0.0 <= table_x <= 1.0 and 0.0 <= table_y <= 1.0
        rows.append(row)
    return rows


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


def table_bounds_svg(plot: PlotArea) -> list[str]:
    lines = []
    for value in (0.0, 1.0):
        y = scale_y(value, plot, -0.25, 1.25)
        lines.append(f'<line x1="{plot.left}" y1="{y:.2f}" x2="{plot.right}" y2="{y:.2f}" stroke="#0f172a" stroke-width="1.5" stroke-dasharray="6 4" opacity="0.7"/>')
    return lines


def table_polyline_segments_svg(
    points: list[dict[str, Any]],
    attr: str,
    color: str,
    plot: PlotArea,
    min_frame: int,
    max_frame: int,
) -> list[str]:
    lines = []
    for segment in table_polyline_segments(points, attr):
        coords = [
            f"{scale_x(int(point['frame']), plot, min_frame, max_frame):.2f},{scale_y(float(point[attr]), plot, -0.25, 1.25):.2f}"
            for point in segment
        ]
        if len(coords) == 1:
            cx, cy = coords[0].split(",", maxsplit=1)
            lines.append(f'<circle cx="{cx}" cy="{cy}" r="2" fill="{color}"/>')
        else:
            lines.append(f'<polyline points="{" ".join(coords)}" fill="none" stroke="{color}" stroke-width="2"/>')
    return lines


def table_polyline_segments(points: list[dict[str, Any]], attr: str) -> list[list[dict[str, Any]]]:
    segments: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for point in points:
        if point.get("detected") and point.get(attr) is not None:
            current.append(point)
            continue
        if current:
            segments.append(current)
            current = []
    if current:
        segments.append(current)
    return segments


def table_inside_markers_svg(points: list[dict[str, Any]], plot: PlotArea, min_frame: int, max_frame: int) -> list[str]:
    lines = []
    y = plot.bottom - 8
    for point in points:
        if not point.get("inside"):
            continue
        x = scale_x(int(point["frame"]), plot, min_frame, max_frame)
        lines.append(f'<circle cx="{x:.2f}" cy="{y}" r="1.8" fill="#22c55e" opacity="0.55"/>')
    return lines


def legend_svg(plot: PlotArea, *, include_table: bool) -> list[str]:
    y = plot.bottom + 44
    entries = [
        ("x", COLORS["x"], "solid"),
        ("y", COLORS["y"], "solid"),
        ("pred bounce/hit", COLORS["bounce"], "solid"),
        ("ref event", COLORS["hit"], "dash"),
        ("missing ball", COLORS["missing"], "solid"),
    ]
    if include_table:
        entries.extend(
            [
                ("table_x", "#0ea5e9", "solid"),
                ("table_y", "#f59e0b", "solid"),
                ("inside table", "#22c55e", "dot"),
            ]
        )
    lines = []
    x = plot.left
    for label, color, style in entries:
        dash = ' stroke-dasharray="4 4"' if style == "dash" else ""
        if style == "dot":
            lines.append(f'<circle cx="{x + 12}" cy="{y}" r="3" fill="{color}" opacity="0.7"/>')
        else:
            lines.append(f'<line x1="{x}" y1="{y}" x2="{x + 24}" y2="{y}" stroke="{color}" stroke-width="2"{dash}/>')
        lines.append(f'<text x="{x + 30}" y="{y + 4}" font-family="sans-serif" font-size="11" fill="#334155">{escape(label)}</text>')
        x += 130
    return lines


def scale_x(frame: float, plot: PlotArea, min_frame: int, max_frame: int) -> float:
    return plot.left + (frame - min_frame) / (max_frame - min_frame) * plot.width


def scale_y(value: float, plot: PlotArea, y_min: float, y_max: float) -> float:
    return plot.bottom - (value - y_min) / (y_max - y_min) * plot.height


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


if __name__ == "__main__":
    raise SystemExit(main())
