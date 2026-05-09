from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tasks/evaluate_event_detection"))

from datapingpong.events.io import load_reference_events
from run import f1, load_predicted_events, match_frames, safe_div


DETAIL_FIELDS = [
    "status",
    "event",
    "item",
    "frame",
    "matched_frame",
    "frame_error",
    "nearest_reference_frame",
    "nearest_reference_delta",
    "nearest_prediction_frame",
    "nearest_prediction_delta",
    "probability",
    "x",
    "y",
    "detected",
    "confidence",
    "bounce_probability",
    "hit_probability",
    "speed",
    "acceleration",
    "turn_angle_degrees",
    "local_x_span",
    "local_y_span",
    "local_detections",
    "local_before_x_displacement",
    "local_after_x_displacement",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Write detailed TP/FP/FN diagnostics for predicted events.")
    parser.add_argument("--reference", type=Path, default=ROOT / "data/annotations/events/bounce_and_hit_frames.json")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=ROOT / "outputs/detect_events_from_ball/local_table_prior_motion_x30_y10_events.json",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/diagnose_event_detection")
    parser.add_argument("--tolerance", type=int, default=4)
    parser.add_argument("--events", nargs="+", default=["bounce", "hit"])
    args = parser.parse_args()

    event_names = set(args.events)
    reference = [row for row in load_reference_events(args.reference) if row["event"] in event_names]
    predictions = [row for row in load_predicted_events_with_payload(args.predictions) if row["event"] in event_names]
    frame_context = load_frame_context(args.predictions)

    details = diagnose(reference, predictions, frame_context, tolerance=args.tolerance, event_names=args.events)
    summary = summarize_details(details, event_names=args.events, tolerance=args.tolerance)
    summary["reference"] = str(args.reference)
    summary["predictions"] = str(args.predictions)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    details_json_path = args.output_dir / "details.json"
    details_csv_path = args.output_dir / "details.csv"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    details_json_path.write_text(json.dumps(details, indent=2), encoding="utf-8")
    write_details_csv(details_csv_path, details)

    print(json.dumps(summary, indent=2))
    print(f"summary: {summary_path}")
    print(f"details: {details_json_path}")
    print(f"csv: {details_csv_path}")
    return 0


def load_predicted_events_with_payload(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "predicted_events" in payload:
        return [
            {
                **row,
                "event": str(row["event"]),
                "frame": int(row["frame"]),
                "item": str(row.get("item", "__default__")),
            }
            for row in payload["predicted_events"]
        ]
    return load_predicted_events(path)


def load_frame_context(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("frames"), list):
        return {}
    rows = {}
    for row in payload["frames"]:
        item = str(row.get("item", "__default__"))
        rows[(item, int(row["frame"]))] = row
    return rows


def diagnose(
    reference: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    frame_context: dict[tuple[str, int], dict[str, Any]],
    *,
    tolerance: int,
    event_names: list[str],
) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for event in event_names:
        items = sorted(
            {
                str(row.get("item", "__default__"))
                for row in reference + predictions
                if row["event"] == event
            }
        )
        for item in items:
            ref_rows = rows_for(reference, event=event, item=item)
            pred_rows = rows_for(predictions, event=event, item=item)
            ref_frames = [int(row["frame"]) for row in ref_rows]
            pred_frames = [int(row["frame"]) for row in pred_rows]
            pred_by_frame = {int(row["frame"]): row for row in pred_rows}
            matches, false_positive, false_negative = match_frames(ref_frames, pred_frames, tolerance)

            for ref_frame, pred_frame in matches:
                details.append(
                    make_detail_row(
                        status="true_positive",
                        event=event,
                        item=item,
                        frame=pred_frame,
                        matched_frame=ref_frame,
                        prediction=pred_by_frame[pred_frame],
                        context=frame_context.get((item, pred_frame)),
                        reference_frames=ref_frames,
                        prediction_frames=pred_frames,
                    )
                )
            for pred_frame in false_positive:
                details.append(
                    make_detail_row(
                        status="false_positive",
                        event=event,
                        item=item,
                        frame=pred_frame,
                        matched_frame=None,
                        prediction=pred_by_frame[pred_frame],
                        context=frame_context.get((item, pred_frame)),
                        reference_frames=ref_frames,
                        prediction_frames=pred_frames,
                    )
                )
            for ref_frame in false_negative:
                context = nearest_context(frame_context, item=item, frame=ref_frame)
                details.append(
                    make_detail_row(
                        status="false_negative",
                        event=event,
                        item=item,
                        frame=ref_frame,
                        matched_frame=None,
                        prediction=None,
                        context=context,
                        reference_frames=ref_frames,
                        prediction_frames=pred_frames,
                    )
                )
    return sorted(details, key=lambda row: (row["event"], row["item"], int(row["frame"]), row["status"]))


def rows_for(rows: list[dict[str, Any]], *, event: str, item: str) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row["event"] == event and str(row.get("item", "__default__")) == item
    ]


def make_detail_row(
    *,
    status: str,
    event: str,
    item: str,
    frame: int,
    matched_frame: int | None,
    prediction: dict[str, Any] | None,
    context: dict[str, Any] | None,
    reference_frames: list[int],
    prediction_frames: list[int],
) -> dict[str, Any]:
    source = prediction or {}
    row: dict[str, Any] = {field: None for field in DETAIL_FIELDS}
    row.update(
        {
            "status": status,
            "event": event,
            "item": item,
            "frame": frame,
            "matched_frame": matched_frame,
            "frame_error": abs(frame - matched_frame) if matched_frame is not None else None,
        }
    )
    nearest_ref = nearest_frame(frame, reference_frames)
    nearest_pred = nearest_frame(frame, prediction_frames)
    row["nearest_reference_frame"] = nearest_ref
    row["nearest_reference_delta"] = abs(frame - nearest_ref) if nearest_ref is not None else None
    row["nearest_prediction_frame"] = nearest_pred
    row["nearest_prediction_delta"] = abs(frame - nearest_pred) if nearest_pred is not None else None

    for key in (
        "probability",
        "x",
        "y",
        "local_x_span",
        "local_y_span",
        "local_detections",
        "local_before_x_displacement",
        "local_after_x_displacement",
    ):
        if key in source:
            row[key] = source[key]
    if context:
        for key in (
            "x",
            "y",
            "detected",
            "confidence",
            "bounce_probability",
            "hit_probability",
            "speed",
            "acceleration",
            "turn_angle_degrees",
        ):
            if key in context and row.get(key) is None:
                row[key] = context[key]
    return row


def nearest_frame(frame: int, candidates: list[int]) -> int | None:
    if not candidates:
        return None
    return min(candidates, key=lambda candidate: (abs(candidate - frame), candidate))


def nearest_context(
    frame_context: dict[tuple[str, int], dict[str, Any]],
    *,
    item: str,
    frame: int,
) -> dict[str, Any] | None:
    frames = [candidate_frame for candidate_item, candidate_frame in frame_context if candidate_item == item]
    nearest = nearest_frame(frame, frames)
    if nearest is None:
        return None
    return frame_context[(item, nearest)]


def summarize_details(
    details: list[dict[str, Any]],
    *,
    event_names: list[str],
    tolerance: int,
) -> dict[str, Any]:
    events: dict[str, Any] = {}
    totals = {"true_positive": 0, "false_positive": 0, "false_negative": 0}
    for event in event_names:
        event_rows = [row for row in details if row["event"] == event]
        true_positive = count_status(event_rows, "true_positive")
        false_positive = count_status(event_rows, "false_positive")
        false_negative = count_status(event_rows, "false_negative")
        predicted = true_positive + false_positive
        reference = true_positive + false_negative
        totals["true_positive"] += true_positive
        totals["false_positive"] += false_positive
        totals["false_negative"] += false_negative
        events[event] = {
            "reference": reference,
            "predicted": predicted,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision": safe_div(true_positive, predicted),
            "recall": safe_div(true_positive, reference),
            "f1": f1(true_positive, predicted, reference),
            "mean_abs_frame_error": mean_numeric(event_rows, "frame_error"),
            "false_positive_mean_probability": mean_numeric(
                [row for row in event_rows if row["status"] == "false_positive"],
                "probability",
            ),
            "false_negative_mean_nearest_prediction_delta": mean_numeric(
                [row for row in event_rows if row["status"] == "false_negative"],
                "nearest_prediction_delta",
            ),
            "false_negative_detected_count": sum(
                1
                for row in event_rows
                if row["status"] == "false_negative" and row.get("detected") is True
            ),
        }
    total_predicted = totals["true_positive"] + totals["false_positive"]
    total_reference = totals["true_positive"] + totals["false_negative"]
    return {
        "tolerance": tolerance,
        "events": events,
        "micro": {
            **totals,
            "precision": safe_div(totals["true_positive"], total_predicted),
            "recall": safe_div(totals["true_positive"], total_reference),
            "f1": f1(totals["true_positive"], total_predicted, total_reference),
        },
    }


def count_status(rows: list[dict[str, Any]], status: str) -> int:
    return sum(1 for row in rows if row["status"] == status)


def mean_numeric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return mean(values)


def write_details_csv(path: Path, details: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DETAIL_FIELDS)
        writer.writeheader()
        writer.writerows([{field: row.get(field) for field in DETAIL_FIELDS} for row in details])


if __name__ == "__main__":
    raise SystemExit(main())
