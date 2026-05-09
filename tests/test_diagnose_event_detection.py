from __future__ import annotations

from pathlib import Path
import importlib.util
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DIAGNOSE_SPEC = importlib.util.spec_from_file_location(
    "diagnose_event_detection_run",
    ROOT / "tasks/diagnose_event_detection/run.py",
)
diagnose_run = importlib.util.module_from_spec(DIAGNOSE_SPEC)
assert DIAGNOSE_SPEC.loader is not None
DIAGNOSE_SPEC.loader.exec_module(diagnose_run)


def test_diagnose_event_detection_marks_tp_fp_fn_with_context() -> None:
    reference = [
        {"event": "bounce", "frame": 10, "item": "__default__"},
        {"event": "bounce", "frame": 30, "item": "__default__"},
    ]
    predictions = [
        {"event": "bounce", "frame": 12, "item": "__default__", "probability": 0.8},
        {"event": "bounce", "frame": 50, "item": "__default__", "probability": 0.7},
    ]
    frame_context = {
        ("__default__", 12): {"frame": 12, "detected": True, "bounce_probability": 0.8, "speed": 4.0},
        ("__default__", 30): {"frame": 30, "detected": False, "bounce_probability": 0.2, "speed": None},
        ("__default__", 50): {"frame": 50, "detected": True, "bounce_probability": 0.7, "speed": 2.0},
    }

    details = diagnose_run.diagnose(
        reference,
        predictions,
        frame_context,
        tolerance=4,
        event_names=["bounce"],
    )

    assert [row["status"] for row in details] == ["true_positive", "false_negative", "false_positive"]
    true_positive = details[0]
    assert true_positive["frame"] == 12
    assert true_positive["matched_frame"] == 10
    assert true_positive["frame_error"] == 2
    false_negative = details[1]
    assert false_negative["frame"] == 30
    assert false_negative["nearest_prediction_frame"] == 12
    assert false_negative["detected"] is False
    false_positive = details[2]
    assert false_positive["nearest_reference_frame"] == 30
    assert false_positive["probability"] == 0.7


def test_summarize_details_includes_metrics_and_diagnostic_means() -> None:
    details = [
        {"event": "hit", "status": "true_positive", "frame_error": 1, "probability": 0.9},
        {"event": "hit", "status": "false_positive", "frame_error": None, "probability": 0.7},
        {
            "event": "hit",
            "status": "false_negative",
            "frame_error": None,
            "nearest_prediction_delta": 8,
            "detected": True,
        },
    ]

    summary = diagnose_run.summarize_details(details, event_names=["hit"], tolerance=4)

    assert summary["events"]["hit"]["precision"] == 0.5
    assert summary["events"]["hit"]["recall"] == 0.5
    assert summary["events"]["hit"]["f1"] == 0.5
    assert summary["events"]["hit"]["mean_abs_frame_error"] == 1
    assert summary["events"]["hit"]["false_positive_mean_probability"] == 0.7
    assert summary["events"]["hit"]["false_negative_mean_nearest_prediction_delta"] == 8
    assert summary["events"]["hit"]["false_negative_detected_count"] == 1
