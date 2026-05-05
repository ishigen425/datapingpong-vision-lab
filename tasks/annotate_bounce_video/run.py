from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

import cv2


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from datapingpong.events.features import build_feature_table
from datapingpong.events.io import load_ball_point_groups
from datapingpong.events.ml import SoftmaxRegression
from datapingpong.events.table import load_table_geometry


def main() -> int:
    parser = argparse.ArgumentParser(description="Render BOUND! text near softmax-predicted bounce frames.")
    parser.add_argument("--video", type=Path, default=ROOT / "data/raw/DJI_0056_001-001.MP4")
    parser.add_argument("--ball", type=Path, default=ROOT / "data/annotations/ball_tracking/DJI_0056_001_predictions.json")
    parser.add_argument("--ball-frame-offset", type=int, default=0, help="Integer frame shift applied to input ball coordinates.")
    parser.add_argument("--model", type=Path, default=ROOT / "models/lightweight_events/openttgames_softmax.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/annotate_bounce_video/DJI_0056_001_bound.mp4")
    parser.add_argument("--events-output", type=Path, default=ROOT / "outputs/annotate_bounce_video/bounce_events.json")
    parser.add_argument("--subtitle-output", type=Path, default=ROOT / "outputs/annotate_bounce_video/bounce_events.ass")
    parser.add_argument("--threshold", type=float, default=0.025)
    parser.add_argument("--nms-window", type=int, default=8)
    parser.add_argument("--display-window", type=int, default=4, help="Frames before/after each bounce to show text.")
    parser.add_argument("--spatial-prior-weight", type=float, default=2.25)
    parser.add_argument("--spatial-prior-sigma-x", type=float, default=0.32)
    parser.add_argument("--spatial-prior-sigma-y", type=float, default=0.30)
    parser.add_argument("--table-geometry", type=Path, default=None, help="Optional JSON with table corner annotations.")
    parser.add_argument("--table-margin", type=float, default=0.05, help="Allowed normalized table margin for bounce candidates.")
    parser.add_argument("--ball-frame-width", type=float, default=None, help="Coordinate width for ball predictions.")
    parser.add_argument("--ball-frame-height", type=float, default=None, help="Coordinate height for ball predictions.")
    parser.add_argument("--video-codec", choices=["libx264", "h264_nvenc"], default="libx264")
    parser.add_argument("--crf", type=int, default=20)
    args = parser.parse_args()

    video_info = read_video_info(args.video)
    model = SoftmaxRegression.from_dict(json.loads(args.model.read_text(encoding="utf-8")))
    table_geometry = load_table_geometry(args.table_geometry)
    model_table_geometry = table_geometry if "table_x" in model.feature_names else None
    groups = load_ball_point_groups(args.ball, confidence_threshold=0.5, frame_offset=args.ball_frame_offset)
    if len(groups) != 1:
        raise ValueError(f"Expected one coordinate group for {args.ball}, got {len(groups)}.")
    item, points = next(iter(groups.items()))
    points_by_frame = {point.frame: point for point in points}
    ball_frame_width, ball_frame_height = resolve_ball_frame_size(
        points,
        video_info,
        width=args.ball_frame_width,
        height=args.ball_frame_height,
    )
    table = build_feature_table(
        item,
        points,
        frame_width=ball_frame_width,
        frame_height=ball_frame_height,
        table_geometry=model_table_geometry,
    )
    bounce_class_index = model.classes.index("bounce")
    probabilities = model.predict_proba(features_for_model(table.feature_names, table.features, model.feature_names))[:, bounce_class_index]
    probabilities = apply_spatial_prior(
        table.frames,
        probabilities,
        points_by_frame,
        video_info,
        ball_frame_width=ball_frame_width,
        ball_frame_height=ball_frame_height,
        weight=args.spatial_prior_weight,
        sigma_x=args.spatial_prior_sigma_x,
        sigma_y=args.spatial_prior_sigma_y,
    )
    candidates = [
        (frame, float(probability))
        for frame, probability in zip(table.frames, probabilities, strict=True)
        if probability >= args.threshold
    ]
    if table_geometry is not None:
        candidates = [
            (frame, probability)
            for frame, probability in candidates
            if table_geometry.contains(
                *scale_point_to_video(points_by_frame[frame], ball_frame_width, ball_frame_height, video_info),
                margin=args.table_margin,
            )
        ]
    bounces = nms(candidates, args.nms_window)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.events_output.write_text(
        json.dumps(
            {
                "video": str(args.video),
                "ball": str(args.ball),
                "model": str(args.model),
                "threshold": args.threshold,
                "nms_window": args.nms_window,
                "display_window": args.display_window,
                "spatial_prior_weight": args.spatial_prior_weight,
                "spatial_prior_sigma_x": args.spatial_prior_sigma_x,
                "spatial_prior_sigma_y": args.spatial_prior_sigma_y,
                "table_geometry": str(args.table_geometry) if args.table_geometry else None,
                "table_margin": args.table_margin,
                "model_uses_table_geometry": model_table_geometry is not None,
                "ball_frame_offset": args.ball_frame_offset,
                "ball_frame_width": ball_frame_width,
                "ball_frame_height": ball_frame_height,
                "video_codec": args.video_codec,
                "fps": video_info["fps"],
                "predicted_events": [
                    {"event": "bounce", "frame": frame, "probability": round(probability, 6)}
                    for frame, probability in bounces
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_ass(args.subtitle_output, bounces, video_info, args.display_window)
    render_video(args.video, args.subtitle_output, args.output, codec=args.video_codec, crf=args.crf)

    print(
        json.dumps(
            {
                "video": str(args.video),
                "output": str(args.output),
                "events_output": str(args.events_output),
                "subtitle_output": str(args.subtitle_output),
                "fps": video_info["fps"],
                "bounce_events": len(bounces),
                "video_codec": args.video_codec,
                "first_frames": [frame for frame, _ in bounces[:10]],
            },
            indent=2,
        )
    )
    return 0


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


def nms(candidates: list[tuple[int, float]], window: int) -> list[tuple[int, float]]:
    selected: list[tuple[int, float]] = []
    for frame, probability in sorted(candidates, key=lambda row: row[1], reverse=True):
        if all(abs(frame - selected_frame) > window for selected_frame, _ in selected):
            selected.append((frame, probability))
    return sorted(selected)


def features_for_model(feature_names: list[str], features: Any, model_feature_names: list[str]) -> Any:
    if feature_names == model_feature_names:
        return features
    feature_index = {name: index for index, name in enumerate(feature_names)}
    missing = [name for name in model_feature_names if name not in feature_index]
    if missing:
        raise ValueError(f"Feature table is missing model features: {missing}")
    return features[:, [feature_index[name] for name in model_feature_names]]


def apply_spatial_prior(
    frames: list[int],
    probabilities: Any,
    points_by_frame: dict[int, Any],
    video_info: dict[str, Any],
    *,
    ball_frame_width: float,
    ball_frame_height: float,
    weight: float,
    sigma_x: float,
    sigma_y: float,
) -> list[float]:
    if weight == 0.0:
        return [float(probability) for probability in probabilities]
    width = float(video_info["width"])
    height = float(video_info["height"])
    adjusted: list[float] = []
    for frame, probability in zip(frames, probabilities, strict=True):
        point = points_by_frame[frame]
        x, y = scale_point_to_video(point, ball_frame_width, ball_frame_height, video_info)
        x_norm = x / width
        y_norm = y / height
        prior = math.exp(
            -0.5 * (((x_norm - 0.5) / sigma_x) ** 2 + ((y_norm - 0.5) / sigma_y) ** 2)
        )
        adjusted.append(_sigmoid(_logit(float(probability)) + weight * prior))
    return adjusted


def resolve_ball_frame_size(
    points: list[Any],
    video_info: dict[str, Any],
    *,
    width: float | None,
    height: float | None,
) -> tuple[float, float]:
    if width is not None and height is not None:
        return float(width), float(height)
    max_x = max(float(point.x or 0.0) for point in points)
    max_y = max(float(point.y or 0.0) for point in points)
    video_width = float(video_info["width"])
    video_height = float(video_info["height"])
    if max_x <= video_width / 2.0 + 2.0 and max_y <= video_height / 2.0 + 2.0:
        return video_width / 3.0, video_height / 3.0
    return video_width, video_height


def scale_point_to_video(
    point: Any,
    ball_frame_width: float,
    ball_frame_height: float,
    video_info: dict[str, Any],
) -> tuple[float | None, float | None]:
    if point.x is None or point.y is None:
        return None, None
    return (
        float(point.x) * float(video_info["width"]) / ball_frame_width,
        float(point.y) * float(video_info["height"]) / ball_frame_height,
    )


def _logit(probability: float) -> float:
    clipped = min(max(probability, 1e-6), 1.0 - 1e-6)
    return math.log(clipped / (1.0 - clipped))


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def write_ass(path: Path, bounces: list[tuple[int, float]], video_info: dict[str, Any], display_window: int) -> None:
    width = video_info["width"]
    height = video_info["height"]
    fps = video_info["fps"]
    font_size = max(48, int(height * 0.08))
    y = int(height * 0.18)
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Bound,Arial,{font_size},&H0000D7FF,&H0000FFFF,&H00000000,&H80000000,"
        "1,0,0,0,100,100,0,0,1,4,1,8,20,20,20,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    for frame, probability in bounces:
        start_frame = max(0, frame - display_window)
        end_frame = min(video_info["frames"], frame + display_window + 1)
        text = f"{{\\pos({width // 2},{y})}}BOUND!  f={frame}  p={probability:.2f}"
        lines.append(f"Dialogue: 0,{ass_time(start_frame / fps)},{ass_time(end_frame / fps)},Bound,,0,0,0,,{text}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def ass_time(seconds: float) -> str:
    centiseconds = int(round(seconds * 100.0))
    hours = centiseconds // 360000
    centiseconds %= 360000
    minutes = centiseconds // 6000
    centiseconds %= 6000
    secs = centiseconds // 100
    centiseconds %= 100
    return f"{hours}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"


def render_video(video: Path, subtitle: Path, output: Path, *, codec: str, crf: int) -> None:
    subtitle_arg = shlex.quote(str(subtitle))
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video),
        "-vf",
        f"ass={subtitle_arg}",
        "-c:v",
        codec,
    ]
    if codec == "h264_nvenc":
        command.extend(["-preset", "p4", "-b:v", "0", "-cq:v", str(crf)])
    else:
        command.extend(["-preset", "medium", "-crf", str(crf)])
    command.extend(["-c:a", "copy", str(output)])
    subprocess.run(command, check=True)


if __name__ == "__main__":
    raise SystemExit(main())
