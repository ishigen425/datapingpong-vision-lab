from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from datapingpong.pose import build_pose_feature_table, load_multi_pose_frames, pose_feature_rows_to_dicts, split_pose_tracks


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract downstream-friendly features from pose-estimation JSON.")
    parser.add_argument("--pose", type=Path, default=ROOT / "outputs/estimate_pose/DJI_0056_001_pose.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/extract_pose_features/DJI_0056_001_pose_features.json")
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument("--min-presence", type=float, default=0.0)
    args = parser.parse_args()

    source_payload = json.loads(args.pose.read_text(encoding="utf-8"))
    frames = load_multi_pose_frames(args.pose)
    role_tracks = split_pose_tracks(frames)
    role_tables = {
        role: build_pose_feature_table(track, min_visibility=args.min_visibility, min_presence=args.min_presence)
        for role, track in role_tracks.items()
    }
    role_rows = {role: {row["frame"]: row for row in pose_feature_rows_to_dicts(table)} for role, table in role_tables.items()}
    feature_names = next(iter(role_tables.values())).feature_names if role_tables else []

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "task": "extract_pose_features",
        "pose": str(args.pose),
        "video": source_payload.get("video"),
        "pose_backend": source_payload.get("pose_backend"),
        "fps": source_payload.get("fps"),
        "frame_width": source_payload.get("frame_width"),
        "frame_height": source_payload.get("frame_height"),
        "frame_count": source_payload.get("frame_count"),
        "min_visibility": args.min_visibility,
        "min_presence": args.min_presence,
        "roles": list(role_tracks.keys()),
        "feature_names": feature_names,
        "frames": [
            {
                "frame": frame.frame,
                "timestamp_ms": frame.timestamp_ms,
                "players": {
                    role: {
                        key: value
                        for key, value in role_rows[role][frame.frame].items()
                        if key not in {"frame", "timestamp_ms"}
                    }
                    for role in role_tracks
                },
            }
            for frame in frames
        ],
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "pose": str(args.pose),
                "output": str(args.output),
                "frames": len(frames),
                "roles": list(role_tracks.keys()),
                "feature_count": len(feature_names),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
