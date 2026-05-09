from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import cv2


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from datapingpong.events.io import load_ball_point_groups
from datapingpong.pose import load_multi_pose_frames, visible_landmark


POSE_CONNECTIONS = (
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
)

POSE_POINTS = (
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
)

POSE_ROLE_COLORS = {
    "left": (255, 120, 0),
    "right": (255, 0, 180),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Render ball, bounce/hit events, and pose outputs into one video.")
    parser.add_argument("--video", type=Path, default=ROOT / "data/raw/DJI_0056_001-001.MP4")
    parser.add_argument("--ball", type=Path, default=ROOT / "data/annotations/ball_tracking/DJI_0056_001_predictions.json")
    parser.add_argument("--ball-frame-offset", type=int, default=0, help="Integer frame shift applied to input ball coordinates.")
    parser.add_argument(
        "--events",
        type=Path,
        default=ROOT / "outputs/detect_events_from_ball/local_table_prior_motion_x30_y10_events.json",
    )
    parser.add_argument("--pose", type=Path, default=ROOT / "outputs/estimate_pose/DJI_0056_001_pose.json")
    parser.add_argument(
        "--pose-features",
        type=Path,
        default=ROOT / "outputs/extract_pose_features/DJI_0056_001_pose_features.json",
    )
    parser.add_argument("--rallies", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/annotate_multimodal_video/DJI_0056_001_multimodal.mp4")
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=ROOT / "outputs/annotate_multimodal_video/DJI_0056_001_multimodal_summary.json",
    )
    parser.add_argument("--trail-length", type=int, default=12, help="How many recent detected ball points to show.")
    parser.add_argument("--event-display-window", type=int, default=4, help="Frames before/after each event label is shown.")
    parser.add_argument("--pose-min-visibility", type=float, default=0.5)
    parser.add_argument("--pose-min-presence", type=float, default=0.0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0, help="Use 0 to render the whole video.")
    parser.add_argument("--video-codec", choices=["libx264", "h264_nvenc"], default="libx264")
    parser.add_argument("--crf", type=int, default=20)
    args = parser.parse_args()

    video_info = read_video_info(args.video)
    ball_rows = load_ball_rows(args.ball, video_info, frame_offset=args.ball_frame_offset)
    events = load_events(args.events)
    pose_rows = {frame.frame: frame for frame in load_multi_pose_frames(args.pose)}
    pose_features = load_pose_features(args.pose_features)
    rallies = load_rallies(args.rallies)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    temp_output = args.output.with_suffix(".tmp.mp4")
    if temp_output.exists():
        temp_output.unlink()

    render_video(
        args.video,
        temp_output,
        video_info,
        ball_rows=ball_rows,
        events=events,
        rallies=rallies,
        pose_rows=pose_rows,
        pose_features=pose_features,
        trail_length=args.trail_length,
        event_display_window=args.event_display_window,
        pose_min_visibility=args.pose_min_visibility,
        pose_min_presence=args.pose_min_presence,
        start_frame=args.start_frame,
        max_frames=args.max_frames,
    )
    transcode_video(temp_output, args.video, args.output, codec=args.video_codec, crf=args.crf)
    temp_output.unlink(missing_ok=True)

    rendered_frames = resolve_rendered_frame_count(video_info["frames"], start_frame=args.start_frame, max_frames=args.max_frames)
    summary = {
        "video": str(args.video),
        "ball": str(args.ball),
        "ball_frame_offset": args.ball_frame_offset,
        "events": str(args.events),
        "pose": str(args.pose),
        "pose_features": str(args.pose_features),
        "rallies": str(args.rallies) if args.rallies else None,
        "output": str(args.output),
        "video_codec": args.video_codec,
        "crf": args.crf,
        "trail_length": args.trail_length,
        "event_display_window": args.event_display_window,
        "pose_min_visibility": args.pose_min_visibility,
        "pose_min_presence": args.pose_min_presence,
        "start_frame": args.start_frame,
        "max_frames": args.max_frames,
        "rendered_frames": rendered_frames,
        "event_counts": count_events(events),
        "rally_count": len(rallies),
        "pose_detected_frames": sum(1 for frame in pose_rows.values() if frame.detected),
        "left_pose_frames": sum(1 for frame in pose_rows.values() if any(pose.role == "left" for pose in frame.poses)),
        "right_pose_frames": sum(1 for frame in pose_rows.values() if any(pose.role == "right" for pose in frame.poses)),
    }
    args.summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def render_video(
    input_video: Path,
    output_video: Path,
    video_info: dict[str, Any],
    *,
    ball_rows: dict[int, dict[str, float | bool | None]],
    events: list[dict[str, Any]],
    rallies: list[dict[str, Any]],
    pose_rows: dict[int, Any],
    pose_features: dict[int, dict[str, Any]],
    trail_length: int,
    event_display_window: int,
    pose_min_visibility: float,
    pose_min_presence: float,
    start_frame: int,
    max_frames: int,
) -> None:
    capture = cv2.VideoCapture(str(input_video))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {input_video}")
    if start_frame > 0:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(video_info["fps"]),
        (int(video_info["width"]), int(video_info["height"])),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output video writer: {output_video}")

    frame_index = start_frame - 1
    history: list[tuple[int, int]] = []
    rendered = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame_index += 1
        if max_frames > 0 and rendered >= max_frames:
            break
        rendered += 1

        ball_row = ball_rows.get(frame_index)
        if ball_row is not None:
            point = point_from_ball_row(ball_row, video_info)
            if point is not None:
                history.append(point)
        history = history[-trail_length:]

        draw_ball_trail(frame, history)
        if ball_row is not None:
            draw_unet_marker(frame, unet_point_from_ball_row(ball_row, video_info), ball_row)
            draw_ball_marker(frame, point_from_ball_row(ball_row, video_info), ball_row)
        active_labels = active_event_labels(events, frame_index, window=event_display_window)
        draw_event_labels(frame, active_labels)
        draw_rally_panel(frame, active_rally(rallies, frame_index), active_labels)
        pose_row = pose_rows.get(frame_index)
        if pose_row is not None:
            draw_pose(
                frame,
                pose_row,
                min_visibility=pose_min_visibility,
                min_presence=pose_min_presence,
            )
        feature_row = pose_features.get(frame_index)
        if feature_row is not None:
            draw_pose_feature_panel(frame, feature_text_lines(feature_row))
        draw_frame_counter(frame, frame_index)
        writer.write(frame)

    writer.release()
    capture.release()


def read_video_info(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if fps <= 0.0:
        raise ValueError(f"Could not read FPS from {path}")
    return {"fps": fps, "width": width, "height": height, "frames": frames}


def load_ball_rows(path: Path, video_info: dict[str, Any], *, frame_offset: int = 0) -> dict[int, dict[str, float | bool | None]]:
    groups = load_ball_point_groups(path, confidence_threshold=0.0, frame_offset=frame_offset)
    if len(groups) != 1:
        raise ValueError(f"Expected one coordinate group for {path}, got {len(groups)}.")
    _, points = next(iter(groups.items()))
    raw_rows = load_raw_ball_prediction_rows(path, frame_offset=frame_offset)
    input_size = load_ball_input_size(path)
    scale_width, scale_height = input_size if input_size is not None else resolve_ball_frame_size(points, video_info)
    rows: dict[int, dict[str, float | bool | None]] = {}
    for point in points:
        row: dict[str, float | bool | None] = {
            "x": float(point.x) if point.x is not None else None,
            "y": float(point.y) if point.y is not None else None,
            "detected": bool(point.detected),
            "confidence": float(point.confidence),
            "input_width": scale_width,
            "input_height": scale_height,
        }
        raw_row = raw_rows.get(point.frame)
        if raw_row is not None:
            row.update(
                {
                    "unet_x": raw_float(raw_row.get("unet_x")),
                    "unet_y": raw_float(raw_row.get("unet_y")),
                    "unet_confidence": raw_float(raw_row.get("unet_confidence")) or 0.0,
                    "unet_detected": bool(raw_row.get("unet_detected", False)),
                }
            )
        rows[point.frame] = row
    return rows


def load_raw_ball_prediction_rows(path: Path, *, frame_offset: int) -> dict[int, dict[str, Any]]:
    if path.suffix == ".jsonl":
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("predictions"), list):
        return {int(row.get("frame", index)) + frame_offset: row for index, row in enumerate(payload["predictions"])}
    if isinstance(payload, list):
        return {int(row.get("frame", index)) + frame_offset: row for index, row in enumerate(payload) if isinstance(row, dict)}
    return {}


def load_ball_input_size(path: Path) -> tuple[float, float] | None:
    if path.suffix == ".jsonl":
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("input_size"), dict):
        return None
    width = raw_float(payload["input_size"].get("width"))
    height = raw_float(payload["input_size"].get("height"))
    if width is None or height is None:
        return None
    return width, height


def raw_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def resolve_ball_frame_size(points: list[Any], video_info: dict[str, Any]) -> tuple[float, float]:
    max_x = max(float(point.x or 0.0) for point in points)
    max_y = max(float(point.y or 0.0) for point in points)
    video_width = float(video_info["width"])
    video_height = float(video_info["height"])
    if max_x <= video_width / 2.0 + 2.0 and max_y <= video_height / 2.0 + 2.0:
        return video_width / 3.0, video_height / 3.0
    return video_width, video_height


def load_events(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["predicted_events"] if isinstance(payload, dict) and "predicted_events" in payload else payload
    return sorted(
        [
            {
                "event": str(row["event"]),
                "frame": int(row["frame"]),
                "probability": float(row.get("probability", 0.0)),
            }
            for row in rows
            if row.get("event") in {"bounce", "hit"}
        ],
        key=lambda row: (row["frame"], row["event"]),
    )


def load_pose_features(path: Path) -> dict[int, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["frames"] if isinstance(payload, dict) and "frames" in payload else payload
    return {int(row["frame"]): row for row in rows}


def load_rallies(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["rallies"] if isinstance(payload, dict) and "rallies" in payload else payload
    return sorted(
        [
            {
                "id": int(row["id"]),
                "start_frame": int(row["start_frame"]),
                "end_frame": int(row["end_frame"]),
                "duration_frames": int(row.get("duration_frames", int(row["end_frame"]) - int(row["start_frame"]) + 1)),
                "event_counts": dict(row.get("event_counts", {})),
                "serve_like_start": bool(row.get("serve_like_start", False)),
                "serve_like_score": float(row["serve_like_score"]) if row.get("serve_like_score") is not None else None,
                "toss_like_start": bool(row.get("toss_like_start", False)),
                "toss_rise_px": float(row["toss_rise_px"]) if row.get("toss_rise_px") is not None else None,
                "toss_x_span_px": float(row["toss_x_span_px"]) if row.get("toss_x_span_px") is not None else None,
            }
            for row in rows
        ],
        key=lambda row: row["start_frame"],
    )


def point_from_ball_row(ball_row: dict[str, float | bool | None], video_info: dict[str, Any]) -> tuple[int, int] | None:
    if not ball_row["detected"] or ball_row["x"] is None or ball_row["y"] is None:
        return None
    x = int(round(float(ball_row["x"]) * float(video_info["width"]) / float(ball_row["input_width"])))
    y = int(round(float(ball_row["y"]) * float(video_info["height"]) / float(ball_row["input_height"])))
    return x, y


def unet_point_from_ball_row(ball_row: dict[str, float | bool | None], video_info: dict[str, Any]) -> tuple[int, int] | None:
    if not ball_row.get("unet_detected") or ball_row.get("unet_x") is None or ball_row.get("unet_y") is None:
        return None
    x = int(round(float(ball_row["unet_x"]) * float(video_info["width"]) / float(ball_row["input_width"])))
    y = int(round(float(ball_row["unet_y"]) * float(video_info["height"]) / float(ball_row["input_height"])))
    return x, y


def draw_ball_trail(frame: Any, history: list[tuple[int, int]]) -> None:
    if len(history) < 2:
        return
    for index in range(1, len(history)):
        start = history[index - 1]
        end = history[index]
        thickness = max(1, int(round(1 + 3 * index / len(history))))
        color = (0, min(255, 80 + 12 * index), 255)
        cv2.line(frame, start, end, color, thickness, cv2.LINE_AA)


def draw_ball_marker(frame: Any, point: tuple[int, int] | None, ball_row: dict[str, float | bool | None]) -> None:
    if point is None:
        return
    cv2.circle(frame, point, 6, (0, 220, 255), 2, cv2.LINE_AA)
    cv2.circle(frame, point, 2, (255, 255, 255), -1, cv2.LINE_AA)
    label = f"ball {float(ball_row['confidence']):.2f}"
    cv2.putText(frame, label, (point[0] + 8, point[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def draw_unet_marker(frame: Any, point: tuple[int, int] | None, ball_row: dict[str, float | bool | None]) -> None:
    if point is None:
        return
    cv2.drawMarker(frame, point, (255, 0, 255), cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
    confidence = float(ball_row.get("unet_confidence") or 0.0)
    label = f"unet {confidence:.2f}"
    cv2.putText(frame, label, (point[0] + 8, point[1] + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def active_event_labels(events: list[dict[str, Any]], frame: int, *, window: int) -> list[str]:
    labels = []
    for event in events:
        delta = abs(int(event["frame"]) - frame)
        if delta <= window:
            labels.append(f"{str(event['event']).upper()} f={int(event['frame'])} p={float(event['probability']):.2f}")
    return labels


def active_rally(rallies: list[dict[str, Any]], frame: int) -> dict[str, Any] | None:
    for rally in rallies:
        if int(rally["start_frame"]) <= frame <= int(rally["end_frame"]):
            return rally
    return None


def draw_event_labels(frame: Any, labels: list[str]) -> None:
    if not labels:
        return
    height, width = frame.shape[:2]
    overlay = frame.copy()
    panel_height = 26 * len(labels) + 18
    cv2.rectangle(overlay, (20, 20), (width - 20, 20 + panel_height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0.0, frame)
    for index, label in enumerate(labels):
        event_name = label.split()[0]
        color = (0, 215, 255) if event_name == "BOUNCE" else (50, 255, 50)
        cv2.putText(frame, label, (40, 48 + 26 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)


def draw_rally_panel(frame: Any, rally: dict[str, Any] | None, active_labels: list[str]) -> None:
    height = frame.shape[0]
    x0 = 20
    y0 = height - 182
    x1 = 420
    y1 = height - 20
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0.0, frame)
    if rally is None:
        cv2.putText(frame, "Rally: none", (x0 + 16, y0 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2, cv2.LINE_AA)
        return
    counts = rally.get("event_counts", {})
    bounce_count = int(counts.get("bounce", 0))
    hit_count = int(counts.get("hit", 0))
    cv2.putText(frame, f"Rally #{int(rally['id'])}", (x0 + 16, y0 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        frame,
        f"range {int(rally['start_frame'])}-{int(rally['end_frame'])}  dur={int(rally['duration_frames'])}",
        (x0 + 16, y0 + 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (220, 220, 220),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"events bounce={bounce_count} hit={hit_count}",
        (x0 + 16, y0 + 84),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (220, 220, 220),
        2,
        cv2.LINE_AA,
    )
    serve_text = "yes" if rally.get("serve_like_start") else "no"
    serve_score = rally.get("serve_like_score")
    cv2.putText(
        frame,
        f"serve_like {serve_text}" + (f"  score={float(serve_score):.2f}" if serve_score is not None else ""),
        (x0 + 16, y0 + 112),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (140, 220, 255) if rally.get("serve_like_start") else (180, 180, 180),
        2,
        cv2.LINE_AA,
    )
    toss_text = "yes" if rally.get("toss_like_start") else "no"
    toss_rise = rally.get("toss_rise_px")
    toss_x_span = rally.get("toss_x_span_px")
    toss_suffix = ""
    if toss_rise is not None and toss_x_span is not None:
        toss_suffix = f"  rise={float(toss_rise):.0f} xspan={float(toss_x_span):.0f}"
    cv2.putText(
        frame,
        f"toss_like {toss_text}{toss_suffix}",
        (x0 + 16, y0 + 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 200, 120) if rally.get("toss_like_start") else (180, 180, 180),
        2,
        cv2.LINE_AA,
    )
    current = ", ".join(label.split()[0] for label in active_labels) if active_labels else "-"
    cv2.putText(
        frame,
        f"now {current}",
        (x0 + 16, y0 + 168),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (100, 255, 100),
        2,
        cv2.LINE_AA,
    )


def draw_pose(frame: Any, pose_row: Any, *, min_visibility: float, min_presence: float) -> None:
    for pose in pose_row.poses:
        color = POSE_ROLE_COLORS.get(pose.role, (200, 200, 0))
        for start_name, end_name in POSE_CONNECTIONS:
            start = visible_landmark(pose, start_name, min_visibility=min_visibility, min_presence=min_presence)
            end = visible_landmark(pose, end_name, min_visibility=min_visibility, min_presence=min_presence)
            start_point = scale_pose_point(start, frame)
            end_point = scale_pose_point(end, frame)
            if start_point is None or end_point is None:
                continue
            cv2.line(frame, start_point, end_point, color, 2, cv2.LINE_AA)
        label_point = None
        for name in POSE_POINTS:
            landmark = visible_landmark(pose, name, min_visibility=min_visibility, min_presence=min_presence)
            point = scale_pose_point(landmark, frame)
            if point is None:
                continue
            if label_point is None:
                label_point = point
            cv2.circle(frame, point, 4, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, point, 6, color, 1, cv2.LINE_AA)
        if label_point is not None:
            cv2.putText(frame, pose.role, (label_point[0] + 8, label_point[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def scale_pose_point(landmark: Any, frame: Any) -> tuple[int, int] | None:
    if landmark is None or landmark.x is None or landmark.y is None:
        return None
    height, width = frame.shape[:2]
    x = int(round(float(landmark.x) * width))
    y = int(round(float(landmark.y) * height))
    return x, y


def feature_text_lines(row: dict[str, Any]) -> list[str]:
    players = row.get("players", {})
    lines: list[str] = []
    for role in ("left", "right"):
        player = players.get(role, {})
        lines.extend(
            [
                f"[{role}] detected={float(player.get('pose_detected', 0.0)):.0f} vis={float(player.get('upper_body_visible_ratio', 0.0)):.2f}",
                f"  tilt={float(player.get('torso_tilt_degrees', 0.0)):.1f} L_elbow={float(player.get('left_elbow_angle_degrees', 0.0)):.1f} R_elbow={float(player.get('right_elbow_angle_degrees', 0.0)):.1f}",
                f"  L_vel=({float(player.get('left_wrist_velocity_x', 0.0)):.2f}, {float(player.get('left_wrist_velocity_y', 0.0)):.2f}) R_vel=({float(player.get('right_wrist_velocity_x', 0.0)):.2f}, {float(player.get('right_wrist_velocity_y', 0.0)):.2f})",
            ]
        )
    return lines


def draw_pose_feature_panel(frame: Any, lines: list[str]) -> None:
    if not lines:
        return
    height, width = frame.shape[:2]
    panel_width = min(760, width - 40)
    line_height = 22
    panel_height = 28 + line_height * len(lines)
    x0 = width - panel_width - 20
    y0 = 20
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_width, y0 + panel_height), (8, 8, 8), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0.0, frame)
    cv2.putText(frame, "Pose features", (x0 + 14, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    for index, line in enumerate(lines):
        cv2.putText(frame, line, (x0 + 14, y0 + 48 + index * line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)


def draw_frame_counter(frame: Any, frame_index: int) -> None:
    height = frame.shape[0]
    cv2.putText(frame, f"frame {frame_index}", (24, height - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)


def transcode_video(temp_video: Path, source_video: Path, output_video: Path, *, codec: str, crf: int) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(temp_video),
        "-i",
        str(source_video),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-c:v",
        codec,
    ]
    if codec == "h264_nvenc":
        command.extend(["-preset", "p4", "-b:v", "0", "-cq:v", str(crf)])
    else:
        command.extend(["-preset", "medium", "-crf", str(crf)])
    command.extend(["-c:a", "copy", "-shortest", str(output_video)])
    subprocess.run(command, check=True)


def count_events(events: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in events:
        name = str(event["event"])
        counts[name] = counts.get(name, 0) + 1
    return counts


def resolve_rendered_frame_count(total_frames: int, *, start_frame: int, max_frames: int) -> int:
    remaining = max(0, total_frames - start_frame)
    if max_frames > 0:
        return min(remaining, max_frames)
    return remaining


if __name__ == "__main__":
    raise SystemExit(main())
