from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate rule-based rally proposal decisions against labeled proposal rows.")
    parser.add_argument("--proposals", type=Path, default=ROOT / "outputs/detect_rallies/DJI_0056_001_rally_proposals.json")
    parser.add_argument("--summary-output", type=Path, default=ROOT / "outputs/evaluate_rally_proposals/summary.json")
    parser.add_argument("--details-output", type=Path, default=ROOT / "outputs/evaluate_rally_proposals/details.json")
    parser.add_argument("--positive-labels", nargs="+", default=["true_rally", "serve_miss_or_ace"])
    parser.add_argument("--label-field", type=str, default="label")
    args = parser.parse_args()

    payload = json.loads(args.proposals.read_text(encoding="utf-8"))
    rows = [row for row in payload["proposals"] if row.get(args.label_field) not in {None, "", "unlabeled"}]
    positive_labels = set(args.positive_labels)

    confusion = Counter()
    by_label: dict[str, Counter[str]] = defaultdict(Counter)
    false_positive: list[dict[str, Any]] = []
    false_negative: list[dict[str, Any]] = []
    positive_groups = build_positive_groups(rows, label_field=args.label_field, positive_labels=positive_labels)

    for row in rows:
        truth_positive = str(row[args.label_field]) in positive_labels
        predicted_positive = bool(row["rule_keep"])
        if truth_positive and predicted_positive:
            confusion["tp"] += 1
        elif truth_positive and not predicted_positive:
            confusion["fn"] += 1
            false_negative.append(slim_row(row, label_field=args.label_field))
        elif not truth_positive and predicted_positive:
            confusion["fp"] += 1
            false_positive.append(slim_row(row, label_field=args.label_field))
        else:
            confusion["tn"] += 1
        by_label[str(row[args.label_field])]["kept" if predicted_positive else "rejected"] += 1

    summary = {
        "proposals": str(args.proposals),
        "label_field": args.label_field,
        "positive_labels": sorted(positive_labels),
        "labeled_rows": len(rows),
        "confusion": dict(confusion),
        "precision": safe_div(confusion["tp"], confusion["tp"] + confusion["fp"]),
        "recall": safe_div(confusion["tp"], confusion["tp"] + confusion["fn"]),
        "f1": f1(confusion["tp"], confusion["fp"], confusion["fn"]),
        "by_label": {label: dict(counts) for label, counts in sorted(by_label.items())},
        "false_positive_count": len(false_positive),
        "false_negative_count": len(false_negative),
        "positive_group_count": len(positive_groups),
    }
    details = {
        "false_positive": false_positive,
        "false_negative": false_negative,
        "positive_groups": positive_groups,
    }

    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    args.details_output.parent.mkdir(parents=True, exist_ok=True)
    args.details_output.write_text(json.dumps(details, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"summary: {args.summary_output}")
    print(f"details: {args.details_output}")
    return 0


def slim_row(row: dict[str, Any], *, label_field: str) -> dict[str, Any]:
    return {
        "proposal_id": row["proposal_id"],
        "start_frame": row["start_frame"],
        "end_frame": row["end_frame"],
        "duration_frames": row["duration_frames"],
        "event_count": row["event_count"],
        "bounce_count": row["bounce_count"],
        "hit_count": row["hit_count"],
        "serve_like_start": row["serve_like_start"],
        "serve_like_score": row["serve_like_score"],
        "toss_like_start": row["toss_like_start"],
        "toss_rise_px": row["toss_rise_px"],
        "toss_x_span_px": row["toss_x_span_px"],
        "rule_keep": row["rule_keep"],
        "rule_reject_reason": row["rule_reject_reason"],
        "label": row[label_field],
    }


def build_positive_groups(
    rows: list[dict[str, Any]],
    *,
    label_field: str,
    positive_labels: set[str],
) -> list[dict[str, Any]]:
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get(label_field)) in positive_labels:
            current.append(row)
            continue
        if current:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return [
        {
            "group_id": index,
            "proposal_ids": [int(row["proposal_id"]) for row in group],
            "labels": [str(row[label_field]) for row in group],
            "start_frame": int(group[0]["start_frame"]),
            "end_frame": int(group[-1]["end_frame"]),
            "event_count": int(sum(int(row["event_count"]) for row in group)),
        }
        for index, group in enumerate(groups, start=1)
    ]


def safe_div(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def f1(tp: int, fp: int, fn: int) -> float | None:
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    if precision is None or recall is None or precision + recall == 0.0:
        return None
    return 2.0 * precision * recall / (precision + recall)


if __name__ == "__main__":
    raise SystemExit(main())
