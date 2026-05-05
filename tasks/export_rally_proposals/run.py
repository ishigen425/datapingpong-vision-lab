from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from datapingpong.events import build_rally_proposals
from datapingpong.events.io import load_ball_points, load_reference_events
from datapingpong.events.rally_ml import build_rally_proposal_feature_table


def load_pose_features(path: Path | None) -> dict[int, dict]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["frames"] if isinstance(payload, dict) and "frames" in payload else payload
    return {int(row["frame"]): row for row in rows}


def main() -> int:
    parser = argparse.ArgumentParser(description="Export kept and rejected rally-start proposals with diagnostics.")
    parser.add_argument("--ball", type=Path, default=ROOT / "outputs/detect_ball_legacy/DJI_0056_001_belief_predictions.json")
    parser.add_argument("--events", type=Path, default=ROOT / "outputs/detect_events_from_ball/DJI_0056_001_belief_events.json")
    parser.add_argument("--pose-features", type=Path, default=ROOT / "outputs/extract_pose_features/DJI_0056_001_pose_features.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/detect_rallies/DJI_0056_001_rally_proposals.json")
    parser.add_argument("--ball-confidence-threshold", type=float, default=0.5)
    parser.add_argument("--ball-frame-offset", type=int, default=0)
    args, unknown = parser.parse_known_args()

    points = load_ball_points(args.ball, confidence_threshold=args.ball_confidence_threshold, frame_offset=args.ball_frame_offset)
    events = load_reference_events(args.events)
    pose_features = load_pose_features(args.pose_features)
    proposal_kwargs = parse_forwarded_detect_rallies_args(unknown)
    proposals = build_rally_proposals(points, events, pose_features, **proposal_kwargs)
    table = build_rally_proposal_feature_table(proposals)

    payload = {
        "ball": str(args.ball),
        "events": str(args.events),
        "pose_features": str(args.pose_features),
        "ball_confidence_threshold": args.ball_confidence_threshold,
        "ball_frame_offset": args.ball_frame_offset,
        "proposal_count": len(table.rows),
        "kept_count": sum(1 for row in table.rows if row["rule_keep"]),
        "rejected_count": sum(1 for row in table.rows if not row["rule_keep"]),
        "feature_names": table.feature_names,
        "proposals": table.rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "proposal_count": payload["proposal_count"], "kept_count": payload["kept_count"]}, indent=2))
    return 0


def parse_forwarded_detect_rallies_args(args: list[str]) -> dict[str, int | float]:
    parser = argparse.ArgumentParser(add_help=False)
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
    parsed = parser.parse_args(args)
    return vars(parsed)


if __name__ == "__main__":
    raise SystemExit(main())
