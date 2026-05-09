from __future__ import annotations

from pathlib import Path
import importlib.util
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datapingpong.events import BallPoint
from datapingpong.events.table import table_geometry_from_dict


PLOT_SPEC = importlib.util.spec_from_file_location(
    "plot_ball_timeseries_run",
    ROOT / "tasks/plot_ball_timeseries/run.py",
)
plot_run = importlib.util.module_from_spec(PLOT_SPEC)
assert PLOT_SPEC.loader is not None
PLOT_SPEC.loader.exec_module(plot_run)


def test_polyline_segments_split_at_missing_points() -> None:
    points = [
        BallPoint(0, 10, 20, detected=True),
        BallPoint(1, None, None, detected=False),
        BallPoint(2, 30, 40, detected=True),
        BallPoint(3, 35, 45, detected=True),
    ]

    segments = plot_run.polyline_segments(points, "x")

    assert [[point.frame for point in segment] for segment in segments] == [[0], [2, 3]]


def test_render_timeseries_svg_includes_event_markers_and_focus() -> None:
    points = [
        BallPoint(10, 100, 200),
        BallPoint(11, 110, 190),
        BallPoint(12, 120, 180),
    ]

    svg = plot_run.render_timeseries_svg(
        points=points,
        predicted_events=[{"event": "hit", "frame": 11, "item": "__default__"}],
        reference_events=[{"event": "bounce", "frame": 12, "item": "__default__"}],
        title="sample",
        start_frame=10,
        end_frame=12,
        width=600,
        height=300,
        focus_frame=11,
    )

    assert "<svg" in svg
    assert "sample" in svg
    assert "HP" in svg
    assert "BR" in svg
    assert "#111827" in svg


def test_render_timeseries_svg_includes_table_relative_panel() -> None:
    points = [
        BallPoint(10, 0, 0),
        BallPoint(11, 50, 50),
        BallPoint(12, 100, 100),
    ]
    table_geometry = table_geometry_from_dict(
        {
            "image_width": 100,
            "image_height": 100,
            "corners": {
                "far_left": [0, 0],
                "far_right": [100, 0],
                "near_right": [100, 100],
                "near_left": [0, 100],
            },
        }
    )

    svg = plot_run.render_timeseries_svg(
        points=points,
        predicted_events=[],
        reference_events=[],
        title="table sample",
        start_frame=10,
        end_frame=12,
        width=600,
        height=420,
        table_geometry=table_geometry,
    )

    assert "table-relative coordinates" in svg
    assert "table_x" in svg
    assert "table_y" in svg
    assert "inside table" in svg
