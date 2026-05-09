from __future__ import annotations

import argparse
from collections.abc import Sequence
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from datapingpong.events.features import build_feature_table
from datapingpong.events.io import load_ball_point_groups, load_reference_events
from datapingpong.events.ml import DenseMLPClassifier, StandardScaler
from datapingpong.events.table import load_table_geometry


TRAIN_EVENT_SPEC = importlib.util.spec_from_file_location("train_event_classifier_run", ROOT / "tasks/train_event_classifier/run.py")
train_event_run = importlib.util.module_from_spec(TRAIN_EVENT_SPEC)
assert TRAIN_EVENT_SPEC.loader is not None
TRAIN_EVENT_SPEC.loader.exec_module(train_event_run)
EVALUATE_EVENT_SPEC = importlib.util.spec_from_file_location(
    "evaluate_event_detection_run",
    ROOT / "tasks/evaluate_event_detection/run.py",
)
evaluate_event_run = importlib.util.module_from_spec(EVALUATE_EVENT_SPEC)
assert EVALUATE_EVENT_SPEC.loader is not None
EVALUATE_EVENT_SPEC.loader.exec_module(evaluate_event_run)

DEFAULT_CLASSES = train_event_run.DEFAULT_CLASSES
DEFAULT_TEST_ITEMS = train_event_run.DEFAULT_TEST_ITEMS
DEFAULT_TRAIN_ITEMS = train_event_run.DEFAULT_TRAIN_ITEMS
build_training_matrix = train_event_run.build_training_matrix
group_events = train_event_run.group_events
nms = train_event_run.nms
summarize = evaluate_event_run.summarize


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a heavier PyTorch event classifier from ball-coordinate features.")
    parser.add_argument("--ball", type=Path, default=ROOT / "data/annotations/openttgames/ball_positions.jsonl")
    parser.add_argument("--events", type=Path, default=ROOT / "data/annotations/openttgames/events.jsonl")
    parser.add_argument("--model-output", type=Path, default=ROOT / "models/lightweight_events/openttgames_bounce_detector_torch_mlp.json")
    parser.add_argument("--predictions-output", type=Path, default=ROOT / "outputs/train_torch_event_classifier/bounce_predictions.json")
    parser.add_argument("--summary-output", type=Path, default=ROOT / "outputs/train_torch_event_classifier/bounce_summary.json")
    parser.add_argument("--train-items", nargs="+", default=DEFAULT_TRAIN_ITEMS)
    parser.add_argument("--test-items", nargs="+", default=DEFAULT_TEST_ITEMS)
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES)
    parser.add_argument("--prediction-events", nargs="+", default=["bounce"])
    parser.add_argument("--positive-radius", type=int, default=2)
    parser.add_argument("--negative-margin", type=int, default=12)
    parser.add_argument("--negative-ratio", type=float, default=1.5)
    parser.add_argument("--frame-width", type=float, default=1280.0)
    parser.add_argument("--frame-height", type=float, default=720.0)
    parser.add_argument("--table-geometry", type=Path, default=None)
    parser.add_argument("--architecture", choices=["mlp", "transformer"], default="mlp")
    parser.add_argument("--hidden-units", nargs="+", type=int, default=[128, 64])
    parser.add_argument("--sequence-radius", type=int, default=6)
    parser.add_argument("--transformer-d-model", type=int, default=64)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-feedforward", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70])
    parser.add_argument("--nms-window", type=int, default=8)
    parser.add_argument("--tolerance", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    if set(args.prediction_events) - (set(args.classes) - {"none"}):
        raise ValueError("--prediction-events must be trained non-none classes")

    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    groups = load_ball_point_groups(args.ball, confidence_threshold=0.0)
    events = load_reference_events(args.events)
    table_geometry = load_table_geometry(args.table_geometry)
    event_by_item = group_events(events, allowed_classes=set(args.classes) - {"none"})
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
    if args.architecture == "transformer":
        x_train, y_train, feature_names = build_sequence_training_matrix(
            feature_tables,
            event_by_item,
            train_items=args.train_items,
            classes=args.classes,
            positive_radius=args.positive_radius,
            negative_margin=args.negative_margin,
            negative_ratio=args.negative_ratio,
            sequence_radius=args.sequence_radius,
            seed=args.seed,
        )
        scaler = StandardScaler.fit(x_train.reshape(-1, x_train.shape[-1]))
        x_scaled = scaler.transform(x_train.reshape(-1, x_train.shape[-1])).reshape(x_train.shape).astype(np.float32)
        network: nn.Module = TorchTransformerClassifier(
            input_size=x_scaled.shape[-1],
            sequence_length=x_scaled.shape[1],
            d_model=args.transformer_d_model,
            heads=args.transformer_heads,
            layers=args.transformer_layers,
            feedforward=args.transformer_feedforward,
            output_size=len(args.classes),
            dropout=args.dropout,
        ).to(device)
    else:
        x_train, y_train, feature_names = build_training_matrix(
            feature_tables,
            event_by_item,
            train_items=args.train_items,
            classes=args.classes,
            positive_radius=args.positive_radius,
            negative_margin=args.negative_margin,
            negative_ratio=args.negative_ratio,
            seed=args.seed,
        )
        scaler = StandardScaler.fit(x_train)
        x_scaled = scaler.transform(x_train).astype(np.float32)
        network = TorchMLP(
            input_size=x_scaled.shape[1],
            hidden_units=args.hidden_units,
            output_size=len(args.classes),
            dropout=args.dropout,
        ).to(device)
    y_train = y_train.astype(np.int64)

    train_network(
        network,
        x_scaled,
        y_train,
        class_weights=balanced_class_weights(y_train, len(args.classes)),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=device,
    )
    if args.architecture == "transformer":
        model = None
        model_payload = export_transformer_payload(
            network,
            scaler=scaler,
            classes=args.classes,
            feature_names=feature_names,
            sequence_radius=args.sequence_radius,
            args=args,
        )
    else:
        model = export_model(network, scaler=scaler, classes=args.classes, feature_names=feature_names)
        model_payload = model.to_dict()

    reference = [row for row in events if row["item"] in set(args.test_items) and row["event"] in set(args.prediction_events)]
    sweep = []
    best_summary = None
    best_predictions = None
    for threshold in args.thresholds:
        if args.architecture == "transformer":
            predictions = predict_events_with_torch_sequence(
                network,
                scaler,
                feature_tables,
                args.test_items,
                prediction_events=args.prediction_events,
                threshold=threshold,
                nms_window=args.nms_window,
                sequence_radius=args.sequence_radius,
                classes=args.classes,
                device=device,
            )
        else:
            assert model is not None
            predictions = predict_events(
                model,
                feature_tables,
                args.test_items,
                prediction_events=args.prediction_events,
                threshold=threshold,
                nms_window=args.nms_window,
            )
        threshold_summary = summarize(reference, predictions, tolerance=args.tolerance, event_names=args.prediction_events)
        micro = threshold_summary["micro"]
        sweep.append({"threshold": threshold, **micro})
        if best_summary is None or score_key(micro) > score_key(best_summary["micro"]):
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
            "classes": args.classes,
            "prediction_events": args.prediction_events,
            "training_rows": int(x_train.shape[0]),
            "sequence_length": int(x_train.shape[1]) if args.architecture == "transformer" else None,
            "feature_count": int(x_train.shape[-1]) if args.architecture == "transformer" else int(x_train.shape[1]),
            "frame_width": args.frame_width,
            "frame_height": args.frame_height,
            "table_geometry": str(args.table_geometry) if args.table_geometry else None,
            "model_type": f"torch_{args.architecture}",
            "hidden_units": args.hidden_units if args.architecture == "mlp" else None,
            "sequence_radius": args.sequence_radius if args.architecture == "transformer" else None,
            "transformer_d_model": args.transformer_d_model if args.architecture == "transformer" else None,
            "transformer_heads": args.transformer_heads if args.architecture == "transformer" else None,
            "transformer_layers": args.transformer_layers if args.architecture == "transformer" else None,
            "transformer_feedforward": args.transformer_feedforward if args.architecture == "transformer" else None,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "device": str(device),
            "best_threshold": best_threshold,
            "threshold_sweep": sweep,
            "nms_window": args.nms_window,
        }
    )

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    args.model_output.write_text(json.dumps(model_payload, indent=2), encoding="utf-8")
    args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
    args.predictions_output.write_text(json.dumps({"predicted_events": best_predictions}, indent=2), encoding="utf-8")
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(best_summary, indent=2), encoding="utf-8")

    print(json.dumps(best_summary, indent=2))
    print(f"model: {args.model_output}")
    print(f"predictions: {args.predictions_output}")
    print(f"summary: {args.summary_output}")
    return 0


class TorchMLP(nn.Module):
    def __init__(self, *, input_size: int, hidden_units: Sequence[int], output_size: int, dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous_size = input_size
        for hidden_size in hidden_units:
            layers.append(nn.Linear(previous_size, hidden_size))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            previous_size = hidden_size
        layers.append(nn.Linear(previous_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class TorchTransformerClassifier(nn.Module):
    def __init__(
        self,
        *,
        input_size: int,
        sequence_length: int,
        d_model: int,
        heads: int,
        layers: int,
        feedforward: int,
        output_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.position_embedding = nn.Parameter(torch.zeros(sequence_length, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.classifier = nn.Linear(d_model, output_size)
        self.center_index = sequence_length // 2

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(values) + self.position_embedding.unsqueeze(0)
        encoded = self.encoder(hidden)
        return self.classifier(encoded[:, self.center_index, :])


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if requested == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_network(
    network: TorchMLP,
    x: np.ndarray,
    y: np.ndarray,
    *,
    class_weights: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
) -> None:
    generator = torch.Generator()
    generator.manual_seed(seed)
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    criterion = nn.CrossEntropyLoss(weight=torch.as_tensor(class_weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(network.parameters(), lr=learning_rate, weight_decay=weight_decay)
    network.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(network(batch_x), batch_y)
            loss.backward()
            optimizer.step()
    network.eval()


def export_model(
    network: TorchMLP,
    *,
    scaler: StandardScaler,
    classes: list[str],
    feature_names: list[str],
) -> DenseMLPClassifier:
    weights = []
    biases = []
    for layer in network.network:
        if isinstance(layer, nn.Linear):
            weights.append(layer.weight.detach().cpu().numpy().T.astype(np.float64))
            biases.append(layer.bias.detach().cpu().numpy().astype(np.float64))
    return DenseMLPClassifier(classes=classes, layer_weights=weights, layer_biases=biases, scaler=scaler, feature_names=feature_names)


def export_transformer_payload(
    network: nn.Module,
    *,
    scaler: StandardScaler,
    classes: list[str],
    feature_names: list[str],
    sequence_radius: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "model_type": "torch_transformer_classifier",
        "classes": classes,
        "feature_names": feature_names,
        "scaler": scaler.to_dict(),
        "sequence_radius": sequence_radius,
        "transformer_d_model": args.transformer_d_model,
        "transformer_heads": args.transformer_heads,
        "transformer_layers": args.transformer_layers,
        "transformer_feedforward": args.transformer_feedforward,
        "dropout": args.dropout,
        "state_dict": {name: value.detach().cpu().tolist() for name, value in network.state_dict().items()},
    }


def build_sequence_training_matrix(
    feature_tables: dict[str, Any],
    events_by_item: dict[str, list[dict[str, Any]]],
    *,
    train_items: list[str],
    classes: list[str],
    positive_radius: int,
    negative_margin: int,
    negative_ratio: float,
    sequence_radius: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    class_to_index = {name: index for index, name in enumerate(classes)}
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
            label_index = class_to_index[event["event"]]
            event_frame = int(event["frame"])
            protected_frames.update(range(event_frame - negative_margin, event_frame + negative_margin + 1))
            for frame in range(event_frame - positive_radius, event_frame + positive_radius + 1):
                index = frame_to_index.get(frame)
                if index is None:
                    continue
                x_rows.append(sequence_for_index(table.features, index, sequence_radius))
                y_rows.append(label_index)
                positive_rows += 1

        negative_candidates = [index for index, frame in enumerate(table.frames) if frame not in protected_frames]
        rng.shuffle(negative_candidates)
        negative_count = min(len(negative_candidates), max(1, int(positive_rows * negative_ratio)))
        for index in negative_candidates[:negative_count]:
            x_rows.append(sequence_for_index(table.features, index, sequence_radius))
            y_rows.append(class_to_index["none"])

    if feature_names is None:
        raise ValueError("No training items were found.")
    return np.stack(x_rows).astype(np.float64), np.asarray(y_rows, dtype=np.int64), feature_names


def sequence_for_index(features: np.ndarray, center_index: int, sequence_radius: int) -> np.ndarray:
    rows = []
    zero = np.zeros(features.shape[1], dtype=np.float64)
    for offset in range(-sequence_radius, sequence_radius + 1):
        index = center_index + offset
        rows.append(features[index] if 0 <= index < features.shape[0] else zero)
    return np.stack(rows)


def balanced_class_weights(y: np.ndarray, class_count: int) -> np.ndarray:
    counts = np.bincount(y, minlength=class_count).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = len(y) / (class_count * counts)
    return (weights / weights.mean()).astype(np.float32)


def predict_events(
    model: DenseMLPClassifier,
    feature_tables: dict[str, Any],
    items: list[str],
    *,
    prediction_events: list[str],
    threshold: float,
    nms_window: int,
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for item in items:
        table = feature_tables[item]
        probabilities = model.predict_proba(table.features)
        for event in prediction_events:
            class_index = model.classes.index(event)
            candidates = [
                (table.frames[row_index], float(probabilities[row_index, class_index]))
                for row_index in range(probabilities.shape[0])
                if probabilities[row_index, class_index] >= threshold
            ]
            predictions.extend(
                {"item": item, "event": event, "frame": frame, "probability": round(probability, 6)}
                for frame, probability in nms(candidates, nms_window)
            )
    return sorted(predictions, key=lambda row: (row["item"], row["frame"], row["event"]))


def predict_events_with_torch_sequence(
    network: nn.Module,
    scaler: StandardScaler,
    feature_tables: dict[str, Any],
    items: list[str],
    *,
    prediction_events: list[str],
    threshold: float,
    nms_window: int,
    sequence_radius: int,
    classes: list[str],
    device: torch.device,
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    network.eval()
    with torch.no_grad():
        for item in items:
            table = feature_tables[item]
            sequences = np.stack([sequence_for_index(table.features, index, sequence_radius) for index in range(table.features.shape[0])])
            scaled = scaler.transform(sequences.reshape(-1, sequences.shape[-1])).reshape(sequences.shape).astype(np.float32)
            logits = network(torch.from_numpy(scaled).to(device))
            probabilities = torch.softmax(logits, dim=1).detach().cpu().numpy()
            for event in prediction_events:
                class_index = classes.index(event)
                candidates = [
                    (table.frames[row_index], float(probabilities[row_index, class_index]))
                    for row_index in range(probabilities.shape[0])
                    if probabilities[row_index, class_index] >= threshold
                ]
                predictions.extend(
                    {"item": item, "event": event, "frame": frame, "probability": round(probability, 6)}
                    for frame, probability in nms(candidates, nms_window)
                )
    return sorted(predictions, key=lambda row: (row["item"], row["frame"], row["event"]))


def score_key(summary: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(summary["f1"] or -1.0),
        float(summary["precision"] or -1.0),
        float(summary["recall"] or -1.0),
    )


if __name__ == "__main__":
    raise SystemExit(main())
