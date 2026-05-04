from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from datapingpong.events.io import load_reference_events


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate predicted bounce/hit events against frame annotations.")
    parser.add_argument("--reference", type=Path, default=ROOT / "data/annotations/events/bounce_and_hit_frames.json")
    parser.add_argument("--predictions", type=Path, default=ROOT / "outputs/detect_events_from_ball/events.json")
    parser.add_argument("--summary", type=Path, default=ROOT / "outputs/evaluate_event_detection/summary.json")
    parser.add_argument("--tolerance", type=int, default=4)
    parser.add_argument("--events", nargs="+", default=["bounce", "hit"])
    args = parser.parse_args()

    reference = [row for row in load_reference_events(args.reference) if row["event"] in set(args.events)]
    predictions = [row for row in load_predicted_events(args.predictions) if row["event"] in set(args.events)]
    summary = summarize(reference, predictions, tolerance=args.tolerance, event_names=args.events)
    summary["reference"] = str(args.reference)
    summary["predictions"] = str(args.predictions)

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"summary: {args.summary}")
    return 0


def load_predicted_events(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "predicted_events" in payload:
        return [
            {"event": str(row["event"]), "frame": int(row["frame"]), "item": str(row.get("item", "__default__"))}
            for row in payload["predicted_events"]
        ]
    if isinstance(payload, list):
        return [
            {"event": str(row["event"]), "frame": int(row["frame"]), "item": str(row.get("item", "__default__"))}
            for row in payload
        ]
    raise ValueError(f"Unsupported predictions format: {path}")


def summarize(
    reference: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    tolerance: int,
    event_names: list[str],
) -> dict[str, Any]:
    by_event: dict[str, Any] = {}
    totals = {"true_positive": 0, "false_positive": 0, "false_negative": 0}
    for event in event_names:
        items = sorted(
            {
                str(row.get("item", "__default__"))
                for row in reference + predictions
                if row["event"] == event
            }
        )
        matches = []
        false_positive = []
        false_negative = []
        ref_count = 0
        pred_count = 0
        for item in items:
            ref_frames = sorted(
                int(row["frame"]) for row in reference if row["event"] == event and str(row.get("item", "__default__")) == item
            )
            pred_frames = sorted(
                int(row["frame"]) for row in predictions if row["event"] == event and str(row.get("item", "__default__")) == item
            )
            ref_count += len(ref_frames)
            pred_count += len(pred_frames)
            item_matches, item_false_positive, item_false_negative = match_frames(ref_frames, pred_frames, tolerance)
            matches.extend(item_matches)
            false_positive.extend(item_false_positive)
            false_negative.extend(item_false_negative)
        errors = [abs(pred - ref) for ref, pred in matches]
        stats = {
            "reference": ref_count,
            "predicted": pred_count,
            "true_positive": len(matches),
            "false_positive": len(false_positive),
            "false_negative": len(false_negative),
            "precision": safe_div(len(matches), pred_count),
            "recall": safe_div(len(matches), ref_count),
            "f1": f1(len(matches), pred_count, ref_count),
            "mean_abs_frame_error": mean(errors) if errors else None,
        }
        by_event[event] = stats
        totals["true_positive"] += stats["true_positive"]
        totals["false_positive"] += stats["false_positive"]
        totals["false_negative"] += stats["false_negative"]

    total_predicted = totals["true_positive"] + totals["false_positive"]
    total_reference = totals["true_positive"] + totals["false_negative"]
    return {
        "tolerance": tolerance,
        "events": by_event,
        "micro": {
            **totals,
            "precision": safe_div(totals["true_positive"], total_predicted),
            "recall": safe_div(totals["true_positive"], total_reference),
            "f1": f1(totals["true_positive"], total_predicted, total_reference),
        },
    }


def match_frames(reference: list[int], predictions: list[int], tolerance: int) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    unmatched_refs = set(range(len(reference)))
    matches: list[tuple[int, int]] = []
    false_positive: list[int] = []
    for prediction in predictions:
        candidates = [
            (abs(prediction - reference[index]), index)
            for index in unmatched_refs
            if abs(prediction - reference[index]) <= tolerance
        ]
        if not candidates:
            false_positive.append(prediction)
            continue
        _, matched_index = min(candidates)
        unmatched_refs.remove(matched_index)
        matches.append((reference[matched_index], prediction))
    false_negative = [reference[index] for index in sorted(unmatched_refs)]
    return matches, false_positive, false_negative


def safe_div(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def f1(true_positive: int, predicted: int, reference: int) -> float | None:
    precision = safe_div(true_positive, predicted)
    recall = safe_div(true_positive, reference)
    if precision is None or recall is None or precision + recall == 0.0:
        return None
    return 2.0 * precision * recall / (precision + recall)


if __name__ == "__main__":
    raise SystemExit(main())
