from __future__ import annotations

from pathlib import Path
import sys
import importlib.util
import json


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datapingpong.events.io import load_ball_points


DETECT_BALL_SPEC = importlib.util.spec_from_file_location("detect_ball_run", ROOT / "tasks/detect_ball_legacy/run.py")
detect_ball_run = importlib.util.module_from_spec(DETECT_BALL_SPEC)
assert DETECT_BALL_SPEC.loader is not None
DETECT_BALL_SPEC.loader.exec_module(detect_ball_run)


def test_prediction_frame_index_targets_window_center() -> None:
    assert detect_ball_run.prediction_frame_index(8, window_size=9) == 4
    assert detect_ball_run.prediction_frame_index(19, window_size=9) == 15


def test_load_ball_points_applies_frame_offset(tmp_path: Path) -> None:
    path = tmp_path / "ball.json"
    path.write_text(json.dumps([{"x": 10, "y": 20, "prod": 0.9}], indent=2), encoding="utf-8")

    points = load_ball_points(path, confidence_threshold=0.5, frame_offset=4)

    assert len(points) == 1
    assert points[0].frame == 4
