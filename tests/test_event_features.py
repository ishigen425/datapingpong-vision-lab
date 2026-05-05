from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datapingpong.events.features import build_feature_table
from datapingpong.events.table import table_geometry_from_dict
from datapingpong.events.trajectory import BallPoint


def test_feature_table_is_stable_across_frame_scales() -> None:
    small = [
        BallPoint(0, 100, 100),
        BallPoint(1, 120, 140),
        BallPoint(2, 140, 180),
        BallPoint(3, 160, 140),
        BallPoint(4, 180, 100),
    ]
    large = [BallPoint(point.frame, point.x * 1.5, point.y * 1.5) for point in small]

    small_table = build_feature_table("small", small, frame_width=1280, frame_height=720)
    large_table = build_feature_table("large", large, frame_width=1920, frame_height=1080)

    assert small_table.feature_names == large_table.feature_names
    np.testing.assert_allclose(small_table.features, large_table.features)


def test_feature_table_adds_table_relative_features() -> None:
    geometry = table_geometry_from_dict(
        {
            "corners": {
                "far_left": [100, 100],
                "far_right": [300, 100],
                "near_right": [300, 300],
                "near_left": [100, 300],
            }
        }
    )
    points = [
        BallPoint(0, 180, 180),
        BallPoint(1, 200, 200),
        BallPoint(2, 220, 180),
    ]

    table = build_feature_table("sample", points, table_geometry=geometry)

    assert table.feature_names[-7:] == [
        "table_x",
        "table_y",
        "table_inside",
        "table_outside_distance",
        "table_edge_margin",
        "table_center_distance",
        "table_net_distance",
    ]
    row = table.features[1]
    np.testing.assert_allclose(row[table.feature_names.index("table_x")], 0.5)
    np.testing.assert_allclose(row[table.feature_names.index("table_y")], 0.5)
    assert row[table.feature_names.index("table_inside")] == 1.0
    assert row[table.feature_names.index("table_outside_distance")] == 0.0


def test_table_geometry_contains_with_margin() -> None:
    geometry = table_geometry_from_dict(
        {
            "corners": {
                "far_left": [0, 0],
                "far_right": [100, 0],
                "near_right": [100, 100],
                "near_left": [0, 100],
            }
        }
    )

    assert geometry.contains(50, 50)
    assert not geometry.contains(106, 50, margin=0.05)
    assert geometry.contains(104, 50, margin=0.05)
