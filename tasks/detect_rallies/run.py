from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from datapingpong.events import detect_rallies, rallies_to_dicts
from datapingpong.events.io import load_ball_points, load_reference_events


def load_pose_features(path: Path | None) -> dict[int, dict]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["frames"] if isinstance(payload, dict) and "frames" in payload else payload
    return {int(row["frame"]): row for row in rows}


def main() -> int:
    parser = argparse.ArgumentParser(description="Segment a match into rally frame ranges from ball tracks and events.")
    parser.add_argument(
        "--ball",
        type=Path,
        default=ROOT / "data/annotations/ball_tracking/DJI_0056_001_predictions.json",
        help="Ball coordinates as legacy JSON, detector predictions JSON, OpenTTGames JSON, or normalized JSONL.",
    )
    parser.add_argument(
        "--events",
        type=Path,
        default=ROOT / "outputs/detect_events_from_ball/local_table_prior_motion_x30_y10_events.json",
        help="Event JSON from detect_events_from_ball or reference event annotations.",
    )
    parser.add_argument("--pose-features", type=Path, default=ROOT / "outputs/extract_pose_features/DJI_0056_001_pose_features.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/detect_rallies/rallies.json")
    parser.add_argument("--ball-confidence-threshold", type=float, default=0.5)
    parser.add_argument("--ball-frame-offset", type=int, default=4)
    parser.add_argument("--fps", type=float, default=120.0)
    parser.add_argument("--max-ball-gap", type=int, default=12)
    parser.add_argument("--min-detected-points", type=int, default=12)
    parser.add_argument("--min-span-frames", type=int, default=24)
    parser.add_argument("--pre-padding", type=int, default=12)
    parser.add_argument("--post-padding", type=int, default=12)
    parser.add_argument("--merge-gap", type=int, default=24)
    parser.add_argument("--min-events-for-auto-keep", type=int, default=1)
    parser.add_argument("--no-event-max-detected-points", type=int, default=96)
    parser.add_argument("--no-event-max-span-frames", type=int, default=120)
    parser.add_argument("--low-event-max-events", type=int, default=1)
    parser.add_argument("--low-event-max-detected-points", type=int, default=180)
    parser.add_argument("--low-event-max-span-frames", type=int, default=220)
    parser.add_argument("--serve-lookback-frames", type=int, default=36)
    parser.add_argument("--serve-min-still-frames", type=int, default=12)
    parser.add_argument("--serve-min-score", type=float, default=0.4)
    parser.add_argument("--serve-toss-window-frames", type=int, default=12)
    parser.add_argument("--serve-min-toss-rise-px", type=float, default=12.0)
    parser.add_argument("--serve-max-toss-x-span-px", type=float, default=24.0)
    parser.add_argument("--serve-max-wrist-speed", type=float, default=0.03)
    parser.add_argument("--serve-max-center-speed", type=float, default=0.006)
    parser.add_argument("--serve-min-visible-ratio", type=float, default=0.6)
    parser.add_argument("--bounce-only-max-direction-changes", type=int, default=0)
    parser.add_argument("--bounce-only-max-abs-delta", type=float, default=400.0)
    parser.add_argument("--bounce-only-compact-max-span-x", type=float, default=80.0)
    parser.add_argument("--long-no-event-min-detected-points", type=int, default=120)
    parser.add_argument("--long-no-event-max-span-x", type=float, default=140.0)
    parser.add_argument("--serve-empty-max-detected-points", type=int, default=60)
    parser.add_argument("--serve-empty-max-span-x", type=float, default=80.0)
    parser.add_argument("--hit-only-max-span-x", type=float, default=180.0)
    parser.add_argument("--hit-only-max-span-frames", type=int, default=220)
    parser.add_argument("--continuation-gap", type=int, default=-1)
    args = parser.parse_args()

    points = load_ball_points(
        args.ball,
        confidence_threshold=args.ball_confidence_threshold,
        frame_offset=args.ball_frame_offset,
    )
    events = load_reference_events(args.events)
    pose_features = load_pose_features(args.pose_features)
    rallies = detect_rallies(
        points,
        events,
        pose_features,
        max_ball_gap=args.max_ball_gap,
        min_detected_points=args.min_detected_points,
        min_span_frames=args.min_span_frames,
        pre_padding=args.pre_padding,
        post_padding=args.post_padding,
        merge_gap=args.merge_gap,
        min_events_for_auto_keep=args.min_events_for_auto_keep,
        no_event_max_detected_points=args.no_event_max_detected_points,
        no_event_max_span_frames=args.no_event_max_span_frames,
        low_event_max_events=args.low_event_max_events,
        low_event_max_detected_points=args.low_event_max_detected_points,
        low_event_max_span_frames=args.low_event_max_span_frames,
        serve_lookback_frames=args.serve_lookback_frames,
        serve_min_still_frames=args.serve_min_still_frames,
        serve_min_score=args.serve_min_score,
        serve_toss_window_frames=args.serve_toss_window_frames,
        serve_min_toss_rise_px=args.serve_min_toss_rise_px,
        serve_max_toss_x_span_px=args.serve_max_toss_x_span_px,
        serve_max_wrist_speed=args.serve_max_wrist_speed,
        serve_max_center_speed=args.serve_max_center_speed,
        serve_min_visible_ratio=args.serve_min_visible_ratio,
        bounce_only_max_direction_changes=args.bounce_only_max_direction_changes,
        bounce_only_max_abs_delta=args.bounce_only_max_abs_delta,
        bounce_only_compact_max_span_x=args.bounce_only_compact_max_span_x,
        long_no_event_min_detected_points=args.long_no_event_min_detected_points,
        long_no_event_max_span_x=args.long_no_event_max_span_x,
        serve_empty_max_detected_points=args.serve_empty_max_detected_points,
        serve_empty_max_span_x=args.serve_empty_max_span_x,
        hit_only_max_span_x=args.hit_only_max_span_x,
        hit_only_max_span_frames=args.hit_only_max_span_frames,
        continuation_gap=args.continuation_gap,
    )
    rally_rows = rallies_to_dicts(rallies, fps=args.fps)

    payload = {
        "ball": str(args.ball),
        "events": str(args.events),
        "pose_features": str(args.pose_features),
        "ball_confidence_threshold": args.ball_confidence_threshold,
        "ball_frame_offset": args.ball_frame_offset,
        "fps": args.fps,
        "max_ball_gap": args.max_ball_gap,
        "min_detected_points": args.min_detected_points,
        "min_span_frames": args.min_span_frames,
        "pre_padding": args.pre_padding,
        "post_padding": args.post_padding,
        "merge_gap": args.merge_gap,
        "min_events_for_auto_keep": args.min_events_for_auto_keep,
        "no_event_max_detected_points": args.no_event_max_detected_points,
        "no_event_max_span_frames": args.no_event_max_span_frames,
        "low_event_max_events": args.low_event_max_events,
        "low_event_max_detected_points": args.low_event_max_detected_points,
        "low_event_max_span_frames": args.low_event_max_span_frames,
        "serve_lookback_frames": args.serve_lookback_frames,
        "serve_min_still_frames": args.serve_min_still_frames,
        "serve_min_score": args.serve_min_score,
        "serve_toss_window_frames": args.serve_toss_window_frames,
        "serve_min_toss_rise_px": args.serve_min_toss_rise_px,
        "serve_max_toss_x_span_px": args.serve_max_toss_x_span_px,
        "serve_max_wrist_speed": args.serve_max_wrist_speed,
        "serve_max_center_speed": args.serve_max_center_speed,
        "serve_min_visible_ratio": args.serve_min_visible_ratio,
        "bounce_only_max_direction_changes": args.bounce_only_max_direction_changes,
        "bounce_only_max_abs_delta": args.bounce_only_max_abs_delta,
        "bounce_only_compact_max_span_x": args.bounce_only_compact_max_span_x,
        "long_no_event_min_detected_points": args.long_no_event_min_detected_points,
        "long_no_event_max_span_x": args.long_no_event_max_span_x,
        "serve_empty_max_detected_points": args.serve_empty_max_detected_points,
        "serve_empty_max_span_x": args.serve_empty_max_span_x,
        "hit_only_max_span_x": args.hit_only_max_span_x,
        "hit_only_max_span_frames": args.hit_only_max_span_frames,
        "continuation_gap": args.continuation_gap,
        "rally_count": len(rally_rows),
        "rallies": rally_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "rally_count": len(rally_rows),
                "total_frames": sum(rally["duration_frames"] for rally in rally_rows),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
