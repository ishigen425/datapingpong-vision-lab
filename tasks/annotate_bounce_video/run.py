from __future__ import annotations

import argparse
import json
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Render Bound! text near softmax-predicted bounce frames.")
    parser.add_argument("--video", type=Path, default=ROOT / "data/raw/DJI_0056_001-001.MP4")
    parser.add_argument("--ball", type=Path, default=ROOT / "data/annotations/ball_tracking/DJI_0056_001_predictions.json")
    parser.add_argument("--model", type=Path, default=ROOT / "models/lightweight_events/openttgames_softmax.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/annotate_bounce_video/DJI_0056_001_bound.mp4")
    parser.add_argument("--events-output", type=Path, default=ROOT / "outputs/annotate_bounce_video/bounce_events.json")
    parser.add_argument("--subtitle-output", type=Path, default=ROOT / "outputs/annotate_bounce_video/bounce_events.ass")
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--nms-window", type=int, default=8)
    parser.add_argument("--display-window", type=int, default=4, help="Frames before/after each bounce to show text.")
    parser.add_argument("--crf", type=int, default=20)
    args = parser.parse_args()

    video_info = read_video_info(args.video)
    model = SoftmaxRegression.from_dict(json.loads(args.model.read_text(encoding="utf-8")))
    groups = load_ball_point_groups(args.ball, confidence_threshold=0.5)
    if len(groups) != 1:
        raise ValueError(f"Expected one coordinate group for {args.ball}, got {len(groups)}.")
    item, points = next(iter(groups.items()))
    table = build_feature_table(item, points)
    bounce_class_index = model.classes.index("bounce")
    probabilities = model.predict_proba(table.features)[:, bounce_class_index]
    candidates = [
        (frame, float(probability))
        for frame, probability in zip(table.frames, probabilities, strict=True)
        if probability >= args.threshold
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
    render_video(args.video, args.subtitle_output, args.output, args.crf)

    print(
        json.dumps(
            {
                "video": str(args.video),
                "output": str(args.output),
                "events_output": str(args.events_output),
                "subtitle_output": str(args.subtitle_output),
                "fps": video_info["fps"],
                "bounce_events": len(bounces),
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
        text = f"{{\\pos({width // 2},{y})}}Bound!  f={frame}  p={probability:.2f}"
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


def render_video(video: Path, subtitle: Path, output: Path, crf: int) -> None:
    subtitle_arg = shlex.quote(str(subtitle))
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video),
        "-vf",
        f"ass={subtitle_arg}",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        str(crf),
        "-c:a",
        "copy",
        str(output),
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    raise SystemExit(main())
