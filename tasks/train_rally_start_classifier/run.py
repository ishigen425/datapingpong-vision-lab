from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from datapingpong.events.ml import SoftmaxRegression
from datapingpong.events.rally_ml import labeled_rows_to_matrix


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a lightweight rally-start classifier from labeled proposal rows.")
    parser.add_argument("--proposals", type=Path, default=ROOT / "outputs/detect_rallies/DJI_0056_001_rally_proposals.json")
    parser.add_argument("--model-output", type=Path, default=ROOT / "models/lightweight_events/rally_start_softmax.json")
    parser.add_argument("--predictions-output", type=Path, default=ROOT / "outputs/train_rally_start_classifier/predictions.json")
    parser.add_argument("--summary-output", type=Path, default=ROOT / "outputs/train_rally_start_classifier/summary.json")
    parser.add_argument("--label-field", type=str, default="label")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--epochs", type=int, default=700)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    payload = json.loads(args.proposals.read_text(encoding="utf-8"))
    rows = payload["proposals"]
    feature_names = list(payload["feature_names"])
    x, y, classes = labeled_rows_to_matrix(rows, feature_names, label_field=args.label_field)
    train_indices, eval_indices = split_indices(len(y), train_fraction=args.train_fraction, seed=args.seed)
    model = SoftmaxRegression.fit(
        x[train_indices],
        y[train_indices],
        classes=classes,
        feature_names=feature_names,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        seed=args.seed,
    )
    predictions = model.predict_proba(x)
    predicted = np.argmax(predictions, axis=1)

    labeled_rows = [row for row in rows if row.get(args.label_field) not in {None, "", "unlabeled"}]
    prediction_rows = []
    for row, probs, pred_index, true_index in zip(labeled_rows, predictions, predicted, y, strict=True):
        prediction_rows.append(
            {
                "proposal_id": row["proposal_id"],
                "start_frame": row["start_frame"],
                "rule_keep": row["rule_keep"],
                "true_label": classes[int(true_index)],
                "predicted_label": classes[int(pred_index)],
                "probabilities": {name: round(float(probs[index]), 6) for index, name in enumerate(classes)},
            }
        )

    eval_accuracy = accuracy(predicted[eval_indices], y[eval_indices]) if len(eval_indices) > 0 else None
    summary = {
        "proposals": str(args.proposals),
        "model_output": str(args.model_output),
        "label_field": args.label_field,
        "classes": classes,
        "feature_names": feature_names,
        "labeled_rows": int(len(y)),
        "train_rows": int(len(train_indices)),
        "eval_rows": int(len(eval_indices)),
        "label_counts": dict(Counter(classes[int(value)] for value in y)),
        "train_accuracy": accuracy(predicted[train_indices], y[train_indices]),
        "eval_accuracy": eval_accuracy,
    }

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    args.model_output.write_text(json.dumps(model.to_dict(), indent=2), encoding="utf-8")
    args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
    args.predictions_output.write_text(json.dumps({"predictions": prediction_rows}, indent=2), encoding="utf-8")
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def split_indices(size: int, *, train_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(size)
    rng.shuffle(indices)
    if size < 2:
        return indices, np.asarray([], dtype=np.int64)
    train_size = min(size - 1, max(1, int(round(size * train_fraction))))
    return indices[:train_size], indices[train_size:]


def accuracy(predicted: np.ndarray, truth: np.ndarray) -> float | None:
    if len(truth) == 0:
        return None
    return float((predicted == truth).mean())


if __name__ == "__main__":
    raise SystemExit(main())
