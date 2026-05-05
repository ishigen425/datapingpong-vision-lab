from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median


ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare ball prediction JSON against reference coordinates.")
    parser.add_argument(
        "--reference",
        type=Path,
        default=ROOT / "data/annotations/ball_tracking/DJI_0056_001_predictions.json",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=ROOT / "outputs/detect_ball_legacy/predictions.json",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=ROOT / "outputs/evaluate_ball_predictions/summary.json",
    )
    parser.add_argument(
        "--diff",
        type=Path,
        default=ROOT / "outputs/evaluate_ball_predictions/frame_diffs.csv",
    )
    parser.add_argument("--reference-threshold", type=float, default=0.5)
    parser.add_argument("--prediction-threshold", type=float, default=0.5)
    parser.add_argument(
        "--frame-offset",
        type=int,
        default=-4,
        help="Reference frame offset applied as reference_frame = prediction_frame + frame_offset.",
    )
    args = parser.parse_args()

    reference = load_reference(args.reference, args.reference_threshold)
    predictions_payload = json.loads(args.predictions.read_text(encoding="utf-8"))
    predictions = predictions_payload["predictions"]

    rows = compare(reference, predictions, args.prediction_threshold, args.frame_offset)
    summary = summarize(rows, predictions_payload, args.reference, args.predictions, args.frame_offset)
    write_diff(args.diff, rows)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"summary: {args.summary}")
    print(f"diff: {args.diff}")
    return 0


def load_reference(path: Path, threshold: float) -> dict[int, dict[str, float | int | bool]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    reference: dict[int, dict[str, float | int | bool]] = {}
    for frame, row in enumerate(data):
        confidence = float(row["prod"])
        x = int(row["x"])
        y = int(row["y"])
        reference[frame] = {
            "frame": frame,
            "x": x,
            "y": y,
            "confidence": confidence,
            "detected": confidence >= threshold and not (x == 0 and y == 0),
        }
    return reference


def compare(
    reference: dict[int, dict[str, float | int | bool]],
    predictions: list[dict[str, float | int | None]],
    prediction_threshold: float,
    frame_offset: int,
) -> list[dict[str, float | int | bool | None]]:
    rows: list[dict[str, float | int | bool | None]] = []
    for pred in predictions:
        frame = int(pred["frame"])
        reference_frame = frame + frame_offset
        ref = reference[reference_frame]
        pred_confidence = float(pred["confidence"])
        pred_detected = pred_confidence >= prediction_threshold and pred["x"] is not None and pred["y"] is not None
        ref_detected = bool(ref["detected"])
        distance = None
        dx = None
        dy = None
        if ref_detected and pred_detected:
            dx = int(pred["x"]) - int(ref["x"])
            dy = int(pred["y"]) - int(ref["y"])
            distance = math.hypot(dx, dy)

        rows.append(
            {
                "frame": frame,
                "reference_frame": reference_frame,
                "ref_x": ref["x"] if ref_detected else None,
                "ref_y": ref["y"] if ref_detected else None,
                "ref_confidence": ref["confidence"],
                "ref_detected": ref_detected,
                "pred_x": pred["x"] if pred_detected else None,
                "pred_y": pred["y"] if pred_detected else None,
                "pred_confidence": pred_confidence,
                "pred_detected": pred_detected,
                "dx": dx,
                "dy": dy,
                "distance_px": distance,
            }
        )
    return rows


def summarize(
    rows: list[dict[str, float | int | bool | None]],
    predictions_payload: dict,
    reference_path: Path,
    predictions_path: Path,
    frame_offset: int,
) -> dict:
    both = [row for row in rows if row["ref_detected"] and row["pred_detected"]]
    distances = [float(row["distance_px"]) for row in both if row["distance_px"] is not None]
    ref_positive = [row for row in rows if row["ref_detected"]]
    pred_positive = [row for row in rows if row["pred_detected"]]
    false_negative = [row for row in rows if row["ref_detected"] and not row["pred_detected"]]
    false_positive = [row for row in rows if not row["ref_detected"] and row["pred_detected"]]

    return {
        "reference": str(reference_path),
        "predictions": str(predictions_path),
        "video": predictions_payload.get("video"),
        "start_frame": predictions_payload.get("start_frame"),
        "max_frames": predictions_payload.get("max_frames"),
        "window_size": predictions_payload.get("window_size"),
        "frame_offset": frame_offset,
        "evaluated_rows": len(rows),
        "ref_detected": len(ref_positive),
        "pred_detected": len(pred_positive),
        "matched_detected": len(both),
        "false_negative": len(false_negative),
        "false_positive": len(false_positive),
        "precision": safe_div(len(both), len(pred_positive)),
        "recall": safe_div(len(both), len(ref_positive)),
        "distance_px": {
            "mean": mean(distances) if distances else None,
            "median": median(distances) if distances else None,
            "max": max(distances) if distances else None,
            "within_1px": count_within(distances, 1.0),
            "within_3px": count_within(distances, 3.0),
            "within_5px": count_within(distances, 5.0),
        },
    }


def safe_div(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def count_within(values: list[float], threshold: float) -> int:
    return sum(1 for value in values if value <= threshold)


def write_diff(path: Path, rows: list[dict[str, float | int | bool | None]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
