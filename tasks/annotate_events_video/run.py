from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
from typing import Any

import cv2


ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description="Render bounce and hit labels on a video.")
    parser.add_argument("--video", type=Path, default=ROOT / "data/raw/DJI_0056_001-001.MP4")
    parser.add_argument("--events", type=Path, default=ROOT / "outputs/detect_events_from_ball/local_table_prior_events.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/annotate_events_video/DJI_0056_001_events_table_nvenc.mp4")
    parser.add_argument("--subtitle-output", type=Path, default=ROOT / "outputs/annotate_events_video/events_table_nvenc.ass")
    parser.add_argument("--display-window", type=int, default=4)
    parser.add_argument("--video-codec", choices=["libx264", "h264_nvenc"], default="libx264")
    parser.add_argument("--crf", type=int, default=20)
    args = parser.parse_args()

    video_info = read_video_info(args.video)
    events = load_events(args.events)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.subtitle_output.parent.mkdir(parents=True, exist_ok=True)
    write_ass(args.subtitle_output, events, video_info, args.display_window)
    render_video(args.video, args.subtitle_output, args.output, codec=args.video_codec, crf=args.crf)
    print(
        json.dumps(
            {
                "video": str(args.video),
                "events": str(args.events),
                "output": str(args.output),
                "subtitle_output": str(args.subtitle_output),
                "video_codec": args.video_codec,
                "event_counts": count_events(events),
            },
            indent=2,
        )
    )
    return 0


def load_events(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["predicted_events"] if isinstance(payload, dict) and "predicted_events" in payload else payload
    return sorted(
        [row for row in rows if row.get("event") in {"bounce", "hit"}],
        key=lambda row: (int(row["frame"]), str(row["event"])),
    )


def count_events(events: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in events:
        event = str(row["event"])
        counts[event] = counts.get(event, 0) + 1
    return counts


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


def write_ass(path: Path, events: list[dict[str, Any]], video_info: dict[str, Any], display_window: int) -> None:
    width = video_info["width"]
    height = video_info["height"]
    fps = video_info["fps"]
    font_size = max(44, int(height * 0.065))
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
        f"Style: Bounce,Arial,{font_size},&H0000D7FF,&H0000FFFF,&H00000000,&H80000000,"
        "1,0,0,0,100,100,0,0,1,4,1,8,20,20,20,1",
        f"Style: Hit,Arial,{font_size},&H0020FF20,&H0000FFFF,&H00000000,&H80000000,"
        "1,0,0,0,100,100,0,0,1,4,1,8,20,20,20,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    for row in events:
        event = str(row["event"])
        frame = int(row["frame"])
        probability = float(row.get("probability", 0.0))
        start_frame = max(0, frame - display_window)
        end_frame = min(video_info["frames"], frame + display_window + 1)
        if event == "hit":
            style = "Hit"
            y = int(height * 0.28)
            label = "HIT"
        else:
            style = "Bounce"
            y = int(height * 0.16)
            label = "BOUND"
        text = f"{{\\pos({width // 2},{y})}}{label}!  f={frame}  p={probability:.2f}"
        lines.append(f"Dialogue: 0,{ass_time(start_frame / fps)},{ass_time(end_frame / fps)},{style},,0,0,0,,{text}")
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
