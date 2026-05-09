from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tasks/evaluate_event_detection"))

from datapingpong.events.features import build_feature_table
from datapingpong.events.io import load_ball_point_groups, load_reference_events
from datapingpong.events.ml import SoftmaxRegression
from datapingpong.events.table import load_table_geometry
from run import summarize


DEFAULT_TRAIN_ITEMS = [f"game_{index}" for index in range(1, 6)]
DEFAULT_TEST_ITEMS = [f"test_{index}" for index in range(1, 8)]
CLASSES = ["none", "bounce"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Train and evaluate an OpenTTGames bounce-only classifier.")
    parser.add_argument("--ball", type=Path, default=ROOT / "data/annotations/openttgames/ball_positions.jsonl")
    parser.add_argument("--events", type=Path, default=ROOT / "data/annotations/openttgames/events.jsonl")
    parser.add_argument("--model-output", type=Path, default=ROOT / "models/lightweight_events/openttgames_bounce_softmax.json")
    parser.add_argument("--predictions-output", type=Path, default=ROOT / "outputs/train_bounce_classifier/predictions.json")
    parser.add_argument("--summary-output", type=Path, default=ROOT / "outputs/train_bounce_classifier/summary.json")
    parser.add_argument("--train-items", nargs="+", default=DEFAULT_TRAIN_ITEMS)
    parser.add_argument("--test-items", nargs="+", default=DEFAULT_TEST_ITEMS)
    parser.add_argument("--positive-radius", type=int, default=2)
    parser.add_argument("--hard-negative-events", nargs="+", default=["net_hit"])
    parser.add_argument("--hard-negative-radius", type=int, default=2)
    parser.add_argument("--negative-margin", type=int, default=12)
    parser.add_argument("--negative-ratio", type=float, default=2.0)
    parser.add_argument("--frame-width", type=float, default=1280.0)
    parser.add_argument("--frame-height", type=float, default=720.0)
    parser.add_argument("--table-geometry", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60])
    parser.add_argument("--nms-window", type=int, default=8)
    parser.add_argument("--tolerance", type=int, default=4)
    args = parser.parse_args()

    groups = load_ball_point_groups(args.ball, confidence_threshold=0.0)
    all_events = load_reference_events(args.events)
    events = [row for row in all_events if row["event"] == "bounce"]
    hard_negative_events = [row for row in all_events if row["event"] in set(args.hard_negative_events)]
    table_geometry = load_table_geometry(args.table_geometry)
    events_by_item = group_events(events)
    hard_negative_events_by_item = group_events(hard_negative_events)
    feature_tables = {
        item: build_feature_table(
            item,
            points,
            frame_width=args.frame_width,
            frame_height=args.frame_height,
            table_geometry=table_geometry,
        )
        for item, points in groups.items()
    }

    x_train, y_train, feature_names = build_training_matrix(
        feature_tables,
        events_by_item,
        hard_negative_events_by_item,
        train_items=args.train_items,
        positive_radius=args.positive_radius,
        hard_negative_radius=args.hard_negative_radius,
        negative_margin=args.negative_margin,
        negative_ratio=args.negative_ratio,
        seed=args.seed,
    )
    model = SoftmaxRegression.fit(
        x_train,
        y_train,
        classes=CLASSES,
        feature_names=feature_names,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        seed=args.seed,
    )

    reference = [row for row in events if row["item"] in set(args.test_items)]
    sweep = []
    best_summary = None
    best_predictions = None
    for threshold in args.thresholds:
        predictions = predict_bounces(model, feature_tables, args.test_items, threshold=threshold, nms_window=args.nms_window)
        threshold_summary = summarize(reference, predictions, tolerance=args.tolerance, event_names=["bounce"])
        bounce_summary = threshold_summary["events"]["bounce"]
        sweep.append({"threshold": threshold, **bounce_summary})
        if best_summary is None or score_key(bounce_summary) > score_key(best_summary["events"]["bounce"]):
            best_summary = threshold_summary
            best_predictions = predictions

    assert best_summary is not None and best_predictions is not None
    best_threshold = max(sweep, key=lambda row: (row["f1"] or -1.0, row["precision"] or -1.0, row["recall"] or -1.0))["threshold"]
    best_summary.update(
        {
            "ball": str(args.ball),
            "events_reference": str(args.events),
            "train_items": args.train_items,
            "test_items": args.test_items,
            "classes": CLASSES,
            "training_rows": int(x_train.shape[0]),
            "feature_count": int(x_train.shape[1]),
            "frame_width": args.frame_width,
            "frame_height": args.frame_height,
            "table_geometry": str(args.table_geometry) if args.table_geometry else None,
            "best_threshold": best_threshold,
            "threshold_sweep": sweep,
            "nms_window": args.nms_window,
            "positive_radius": args.positive_radius,
            "hard_negative_events": args.hard_negative_events,
            "hard_negative_radius": args.hard_negative_radius,
            "negative_margin": args.negative_margin,
            "negative_ratio": args.negative_ratio,
        }
    )

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    args.model_output.write_text(json.dumps(model.to_dict(), indent=2), encoding="utf-8")
    args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
    args.predictions_output.write_text(json.dumps({"predicted_events": best_predictions}, indent=2), encoding="utf-8")
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(best_summary, indent=2), encoding="utf-8")

    print(json.dumps(best_summary, indent=2))
    print(f"model: {args.model_output}")
    print(f"predictions: {args.predictions_output}")
    print(f"summary: {args.summary_output}")
    return 0


def group_events(events: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_item: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in events:
        by_item[str(row["item"])].append(row)
    return by_item


def build_training_matrix(
    feature_tables: dict[str, Any],
    events_by_item: dict[str, list[dict[str, Any]]],
    hard_negative_events_by_item: dict[str, list[dict[str, Any]]],
    *,
    train_items: list[str],
    positive_radius: int,
    hard_negative_radius: int,
    negative_margin: int,
    negative_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    x_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    feature_names: list[str] | None = None

    for item in train_items:
        table = feature_tables[item]
        feature_names = table.feature_names
        frame_to_index = {frame: index for index, frame in enumerate(table.frames)}
        protected_frames: set[int] = set()
        positive_rows = 0
        for event in events_by_item[item]:
            event_frame = int(event["frame"])
            protected_frames.update(range(event_frame - negative_margin, event_frame + negative_margin + 1))
            for frame in range(event_frame - positive_radius, event_frame + positive_radius + 1):
                index = frame_to_index.get(frame)
                if index is None:
                    continue
                x_rows.append(table.features[index])
                y_rows.append(CLASSES.index("bounce"))
                positive_rows += 1

        hard_negative_rows = 0
        for event in hard_negative_events_by_item[item]:
            event_frame = int(event["frame"])
            protected_frames.update(range(event_frame - negative_margin, event_frame + negative_margin + 1))
            for frame in range(event_frame - hard_negative_radius, event_frame + hard_negative_radius + 1):
                index = frame_to_index.get(frame)
                if index is None:
                    continue
                x_rows.append(table.features[index])
                y_rows.append(CLASSES.index("none"))
                hard_negative_rows += 1

        negative_candidates = [index for index, frame in enumerate(table.frames) if frame not in protected_frames]
        rng.shuffle(negative_candidates)
        negative_count = min(len(negative_candidates), max(1, int((positive_rows + hard_negative_rows) * negative_ratio)))
        for index in negative_candidates[:negative_count]:
            x_rows.append(table.features[index])
            y_rows.append(CLASSES.index("none"))

    if feature_names is None:
        raise ValueError("No training items were found.")
    return np.vstack(x_rows), np.asarray(y_rows, dtype=np.int64), feature_names


def predict_bounces(
    model: SoftmaxRegression,
    feature_tables: dict[str, Any],
    items: list[str],
    *,
    threshold: float,
    nms_window: int,
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    bounce_index = model.classes.index("bounce")
    for item in items:
        table = feature_tables[item]
        probabilities = model.predict_proba(table.features)
        candidates = [
            (table.frames[row_index], float(probabilities[row_index, bounce_index]))
            for row_index in range(probabilities.shape[0])
            if probabilities[row_index, bounce_index] >= threshold
        ]
        predictions.extend(
            {"item": item, "event": "bounce", "frame": frame, "probability": round(probability, 6)}
            for frame, probability in nms(candidates, nms_window)
        )
    return sorted(predictions, key=lambda row: (row["item"], row["frame"]))


def nms(candidates: list[tuple[int, float]], window: int) -> list[tuple[int, float]]:
    selected: list[tuple[int, float]] = []
    for frame, probability in sorted(candidates, key=lambda row: row[1], reverse=True):
        if all(abs(frame - selected_frame) > window for selected_frame, _ in selected):
            selected.append((frame, probability))
    return sorted(selected)


def score_key(summary: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(summary["f1"] or -1.0),
        float(summary["precision"] or -1.0),
        float(summary["recall"] or -1.0),
    )


if __name__ == "__main__":
    raise SystemExit(main())
