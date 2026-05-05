from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from datapingpong.events import detect_event_peaks, score_trajectory
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
    parser.add_argument("--bounce-threshold", type=float, default=0.35)
    parser.add_argument("--hit-threshold", type=float, default=0.65)
    parser.add_argument("--nms-window", type=int, default=8)
    parser.add_argument("--max-gap", type=int, default=3)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--table-geometry", type=Path, default=None, help="Optional JSON with table corner annotations.")
    parser.add_argument("--table-margin", type=float, default=0.05, help="Allowed normalized table margin for bounce candidates.")
    args = parser.parse_args()

    groups = load_ball_point_groups(args.input, confidence_threshold=args.confidence_threshold)
    table_geometry = load_table_geometry(args.table_geometry)
    probabilities = []
    peaks = []
    for item, points in groups.items():
        item_probabilities = score_trajectory(points, max_gap=args.max_gap, smooth_window=args.smooth_window)
        item_peaks = detect_event_peaks(
            item_probabilities,
            bounce_threshold=args.bounce_threshold,
            hit_threshold=args.hit_threshold,
            nms_window=args.nms_window,
        )
        if table_geometry is not None:
            item_peaks = [
                peak
                for peak in item_peaks
                if peak.event != "bounce" or table_geometry.contains(peak.x, peak.y, margin=args.table_margin)
            ]
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
        "bounce_threshold": args.bounce_threshold,
        "hit_threshold": args.hit_threshold,
        "nms_window": args.nms_window,
        "max_gap": args.max_gap,
        "smooth_window": args.smooth_window,
        "table_geometry": str(args.table_geometry) if args.table_geometry else None,
        "table_margin": args.table_margin,
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


if __name__ == "__main__":
    raise SystemExit(main())
