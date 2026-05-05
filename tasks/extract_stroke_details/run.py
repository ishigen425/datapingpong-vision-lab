from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from datapingpong.events.io import load_ball_points, load_reference_events
from datapingpong.events.stroke import infer_hit_strokes, strokes_to_dicts


def load_pose_features(path: Path | None) -> tuple[dict[int, dict], int, int]:
    if path is None or not path.exists():
        return {}, 1920, 1080
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["frames"] if isinstance(payload, dict) and "frames" in payload else payload
    frame_width = int(payload.get("frame_width", 1920)) if isinstance(payload, dict) else 1920
    frame_height = int(payload.get("frame_height", 1080)) if isinstance(payload, dict) else 1080
    return {int(row["frame"]): row for row in rows}, frame_width, frame_height


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract per-hit stroke details from ball, hit events, and pose features.")
    parser.add_argument("--ball", type=Path, default=ROOT / "outputs/detect_ball_legacy/DJI_0056_001_belief_predictions.json")
    parser.add_argument("--events", type=Path, default=ROOT / "outputs/detect_events_from_ball/DJI_0056_001_belief_events.json")
    parser.add_argument("--pose-features", type=Path, default=ROOT / "outputs/extract_pose_features/DJI_0056_001_pose_features.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/extract_stroke_details/DJI_0056_001_strokes.json")
    parser.add_argument("--ball-confidence-threshold", type=float, default=0.5)
    parser.add_argument("--ball-frame-offset", type=int, default=0)
    parser.add_argument("--window-frames", type=int, default=4)
    parser.add_argument("--left-racket-hand", choices=["left", "right"], default="right")
    parser.add_argument("--right-racket-hand", choices=["left", "right"], default="right")
    args = parser.parse_args()

    points = load_ball_points(args.ball, confidence_threshold=args.ball_confidence_threshold, frame_offset=args.ball_frame_offset)
    events = load_reference_events(args.events)
    pose_features, frame_width, frame_height = load_pose_features(args.pose_features)
    strokes = infer_hit_strokes(
        events,
        points,
        pose_features,
        frame_width=frame_width,
        frame_height=frame_height,
        window_frames=args.window_frames,
        left_racket_hand=args.left_racket_hand,
        right_racket_hand=args.right_racket_hand,
    )
    payload = {
        "ball": str(args.ball),
        "events": str(args.events),
        "pose_features": str(args.pose_features),
        "ball_confidence_threshold": args.ball_confidence_threshold,
        "ball_frame_offset": args.ball_frame_offset,
        "window_frames": args.window_frames,
        "left_racket_hand": args.left_racket_hand,
        "right_racket_hand": args.right_racket_hand,
        "stroke_count": len(strokes),
        "strokes": strokes_to_dicts(strokes),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "stroke_count": len(strokes)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
