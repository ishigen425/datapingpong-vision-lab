from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tasks/evaluate_event_detection"))

from datapingpong.events.io import load_reference_events
from datapingpong.events.ml import SoftmaxRegression
from datapingpong.events.table import load_table_geometry, table_feature_names
from run import match_frames, summarize


DEFAULT_CLASSES = ["none", "bounce"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a local ML filter for rule-generated event candidates.")
    parser.add_argument(
        "--candidate-events",
        type=Path,
        default=ROOT / "outputs/detect_events_from_ball/local_table_prior_motion_x30_y10_events.json",
    )
    parser.add_argument("--reference", type=Path, default=ROOT / "data/annotations/events/bounce_and_hit_frames.json")
    parser.add_argument("--table-geometry", type=Path, default=None)
    parser.add_argument("--model-output", type=Path, default=ROOT / "models/lightweight_events/local_candidate_softmax.json")
    parser.add_argument("--predictions-output", type=Path, default=ROOT / "outputs/train_event_candidate_classifier/predictions.json")
    parser.add_argument("--summary-output", type=Path, default=ROOT / "outputs/train_event_candidate_classifier/summary.json")
    parser.add_argument("--rows-output", type=Path, default=ROOT / "outputs/train_event_candidate_classifier/training_rows.csv")
    parser.add_argument(
        "--classes",
        nargs="+",
        default=DEFAULT_CLASSES,
        help="Classes to train. Use the default bounce-only setup unless hit labels have been manually verified.",
    )
    parser.add_argument("--tolerance", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=1200)
    parser.add_argument("--test-fold", type=int, default=1)
    parser.add_argument("--fold-count", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=None, help="Shared accept threshold. Overrides class-specific thresholds when set.")
    parser.add_argument("--bounce-threshold", type=float, default=0.30)
    parser.add_argument("--hit-threshold", type=float, default=0.15)
    parser.add_argument("--require-argmax", action="store_true", help="Require the model's top class to match the candidate event.")
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    classes = validate_classes(args.classes)
    event_classes = [name for name in classes if name != "none"]

    payload = json.loads(args.candidate_events.read_text(encoding="utf-8"))
    candidates = load_candidates(payload, allowed_events=set(event_classes))
    frame_context = load_frame_context(payload)
    reference = [row for row in load_reference_events(args.reference) if row["event"] in set(event_classes)]
    table_geometry = load_table_geometry(args.table_geometry)

    labeled = label_candidates(candidates, reference, tolerance=args.tolerance)
    feature_names = candidate_feature_names(include_table=table_geometry is not None)
    rows = [
        build_candidate_row(
            candidate,
            frame_context.get((candidate["item"], candidate["frame"])),
            table_geometry=table_geometry,
        )
        for candidate in labeled
    ]
    x = np.asarray([row["features"] for row in rows], dtype=np.float64)
    y = np.asarray([classes.index(candidate["label"]) for candidate in labeled], dtype=np.int64)
    split = np.asarray(
        [
            split_name(candidate["frame"], block_size=args.block_size, fold_count=args.fold_count, test_fold=args.test_fold)
            for candidate in labeled
        ]
    )
    train_mask = split == "train"
    test_mask = split == "test"
    if not train_mask.any() or not test_mask.any():
        raise ValueError("Temporal split produced empty train or test rows.")

    model = SoftmaxRegression.fit(
        x[train_mask],
        y[train_mask],
        classes=classes,
        feature_names=feature_names,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        seed=args.seed,
    )
    probabilities = model.predict_proba(x)
    thresholds = {
        "bounce": args.threshold if args.threshold is not None else args.bounce_threshold,
        "hit": args.threshold if args.threshold is not None else args.hit_threshold,
    }
    predictions = select_predictions(
        labeled,
        probabilities,
        split,
        classes=classes,
        thresholds=thresholds,
        require_argmax=args.require_argmax,
    )
    test_reference = [
        row
        for row in reference
        if split_name(int(row["frame"]), block_size=args.block_size, fold_count=args.fold_count, test_fold=args.test_fold) == "test"
    ]
    summary = summarize(test_reference, predictions, tolerance=args.tolerance, event_names=event_classes)
    raw_test_predictions = [
        {"item": row["item"], "event": row["event"], "frame": int(row["frame"])}
        for index, row in enumerate(labeled)
        if split[index] == "test"
    ]
    raw_candidate_summary = summarize(test_reference, raw_test_predictions, tolerance=args.tolerance, event_names=event_classes)
    summary.update(
        {
            "candidate_events": str(args.candidate_events),
            "reference": str(args.reference),
            "table_geometry": str(args.table_geometry) if args.table_geometry else None,
            "classes": classes,
            "thresholds": thresholds,
            "require_argmax": args.require_argmax,
            "block_size": args.block_size,
            "fold_count": args.fold_count,
            "test_fold": args.test_fold,
            "training_rows": int(train_mask.sum()),
            "test_rows": int(test_mask.sum()),
            "feature_count": len(feature_names),
            "train_label_counts": label_counts(y[train_mask], classes),
            "test_label_counts": label_counts(y[test_mask], classes),
            "raw_candidate_summary": raw_candidate_summary,
            "label_caveat": "Default excludes hit because local DJI hit labels are inferred and not valid supervised labels.",
        }
    )

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    args.model_output.write_text(json.dumps(model.to_dict(), indent=2), encoding="utf-8")
    args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
    args.predictions_output.write_text(json.dumps({"predicted_events": predictions}, indent=2), encoding="utf-8")
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_rows(args.rows_output, labeled, rows, probabilities, split, classes)

    print(json.dumps(summary, indent=2))
    print(f"model: {args.model_output}")
    print(f"predictions: {args.predictions_output}")
    print(f"summary: {args.summary_output}")
    print(f"rows: {args.rows_output}")
    return 0


def validate_classes(classes: list[str]) -> list[str]:
    if not classes or classes[0] != "none":
        raise ValueError("--classes must start with 'none'")
    unsupported = set(classes) - {"none", "bounce", "hit"}
    if unsupported:
        raise ValueError(f"Unsupported classes: {sorted(unsupported)}")
    return classes


def load_candidates(payload: Any, *, allowed_events: set[str]) -> list[dict[str, Any]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("predicted_events"), list):
        raise ValueError("Candidate input must be a detect_events_from_ball JSON payload.")
    rows = []
    for row in payload["predicted_events"]:
        event = str(row["event"])
        if event not in allowed_events:
            continue
        rows.append(
            {
                **row,
                "event": event,
                "frame": int(row["frame"]),
                "item": str(row.get("item", "__default__")),
            }
        )
    return sorted(rows, key=lambda row: (row["item"], row["frame"], row["event"]))


def load_frame_context(payload: Any) -> dict[tuple[str, int], dict[str, Any]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("frames"), list):
        return {}
    rows = {}
    for row in payload["frames"]:
        item = str(row.get("item", "__default__"))
        rows[(item, int(row["frame"]))] = row
    return rows


def label_candidates(candidates: list[dict[str, Any]], reference: list[dict[str, Any]], *, tolerance: int) -> list[dict[str, Any]]:
    labeled = [{**row, "label": "none"} for row in candidates]
    by_key = {(row["item"], row["event"], row["frame"]): row for row in labeled}
    candidate_events = sorted({row["event"] for row in candidates})
    for event in candidate_events:
        items = sorted({row["item"] for row in candidates + reference if row["event"] == event})
        for item in items:
            ref_frames = sorted(int(row["frame"]) for row in reference if row["event"] == event and str(row.get("item", "__default__")) == item)
            pred_frames = sorted(int(row["frame"]) for row in candidates if row["event"] == event and row["item"] == item)
            matches, _, _ = match_frames(ref_frames, pred_frames, tolerance)
            for _, pred_frame in matches:
                by_key[(item, event, pred_frame)]["label"] = event
    return labeled


def candidate_feature_names(*, include_table: bool) -> list[str]:
    names = [
        "candidate_is_bounce",
        "candidate_is_hit",
        "candidate_probability",
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
        "same_direction_x",
        "opposite_direction_x",
        "min_abs_directional_x",
    ]
    if include_table:
        names.extend(table_feature_names())
    return names


def build_candidate_row(
    candidate: dict[str, Any],
    context: dict[str, Any] | None,
    *,
    table_geometry: Any | None,
) -> dict[str, Any]:
    source = {**(context or {}), **candidate}
    before_dx = number(source.get("local_before_x_displacement"))
    after_dx = number(source.get("local_after_x_displacement"))
    same_direction = 1.0 if before_dx is not None and after_dx is not None and before_dx * after_dx > 0.0 else 0.0
    opposite_direction = 1.0 if before_dx is not None and after_dx is not None and before_dx * after_dx < 0.0 else 0.0
    min_abs_directional = min(abs(before_dx), abs(after_dx)) if before_dx is not None and after_dx is not None else 0.0
    x = number(source.get("x"))
    y = number(source.get("y"))
    features = [
        1.0 if candidate["event"] == "bounce" else 0.0,
        1.0 if candidate["event"] == "hit" else 0.0,
        number(source.get("probability"), 0.0),
        number(x, 0.0),
        number(y, 0.0),
        1.0 if source.get("detected") is True else 0.0,
        number(source.get("confidence"), 0.0),
        number(source.get("bounce_probability"), 0.0),
        number(source.get("hit_probability"), 0.0),
        number(source.get("speed"), 0.0),
        number(source.get("acceleration"), 0.0),
        number(source.get("turn_angle_degrees"), 0.0),
        number(source.get("local_x_span"), 0.0),
        number(source.get("local_y_span"), 0.0),
        number(source.get("local_detections"), 0.0),
        number(before_dx, 0.0),
        number(after_dx, 0.0),
        same_direction,
        opposite_direction,
        min_abs_directional,
    ]
    if table_geometry is not None:
        features.extend(table_geometry.features_for_point(x, y))
    return {"features": features}


def select_predictions(
    candidates: list[dict[str, Any]],
    probabilities: np.ndarray,
    split: np.ndarray,
    *,
    classes: list[str],
    thresholds: dict[str, float],
    require_argmax: bool,
) -> list[dict[str, Any]]:
    rows = []
    for index, candidate in enumerate(candidates):
        if split[index] != "test":
            continue
        event = candidate["event"]
        class_index = classes.index(event)
        probability = float(probabilities[index, class_index])
        none_probability = float(probabilities[index, classes.index("none")])
        predicted_class = classes[int(np.argmax(probabilities[index]))]
        if require_argmax and predicted_class != event:
            continue
        if probability < thresholds[event]:
            continue
        rows.append(
            {
                "item": candidate["item"],
                "event": event,
                "frame": int(candidate["frame"]),
                "probability": round(probability, 6),
                "none_probability": round(none_probability, 6),
                "candidate_probability": round(number(candidate.get("probability"), 0.0), 6),
            }
        )
    return sorted(rows, key=lambda row: (row["item"], row["frame"], row["event"]))


def split_name(frame: int, *, block_size: int, fold_count: int, test_fold: int) -> str:
    fold = (frame // block_size) % fold_count
    return "test" if fold == test_fold else "train"


def label_counts(y: np.ndarray, classes: list[str]) -> dict[str, int]:
    counts = np.bincount(y, minlength=len(classes))
    return {name: int(counts[index]) for index, name in enumerate(classes)}


def write_rows(
    path: Path,
    candidates: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    probabilities: np.ndarray,
    split: np.ndarray,
    classes: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["split", "item", "frame", "candidate_event", "label", *[f"p_{name}" for name in classes]]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, candidate in enumerate(candidates):
            writer.writerow(
                {
                    "split": split[index],
                    "item": candidate["item"],
                    "frame": candidate["frame"],
                    "candidate_event": candidate["event"],
                    "label": candidate["label"],
                    **{f"p_{name}": round(float(probabilities[index, class_index]), 6) for class_index, name in enumerate(classes)},
                }
            )


def number(value: Any, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    return float(value)


if __name__ == "__main__":
    raise SystemExit(main())
