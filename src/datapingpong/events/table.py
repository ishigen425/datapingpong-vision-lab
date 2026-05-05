from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


CORNER_NAMES = ("far_left", "far_right", "near_right", "near_left")


@dataclass(frozen=True)
class TableGeometry:
    image_width: float | None
    image_height: float | None
    homography: np.ndarray

    def image_to_table(self, x: float, y: float) -> tuple[float, float]:
        point = self.homography @ np.asarray([x, y, 1.0], dtype=np.float64)
        if point[2] == 0.0:
            return 0.0, 0.0
        return float(point[0] / point[2]), float(point[1] / point[2])

    def features_for_point(self, x: float | None, y: float | None) -> list[float]:
        if x is None or y is None:
            return [0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0]
        table_x, table_y = self.image_to_table(float(x), float(y))
        outside_distance = _outside_unit_square_distance(table_x, table_y)
        inside = 1.0 if outside_distance == 0.0 else 0.0
        edge_margin = min(table_x, 1.0 - table_x, table_y, 1.0 - table_y)
        center_distance = math.hypot(table_x - 0.5, table_y - 0.5)
        net_distance = abs(table_y - 0.5)
        return [table_x, table_y, inside, outside_distance, edge_margin, center_distance, net_distance]

    def contains(self, x: float | None, y: float | None, *, margin: float = 0.0) -> bool:
        if x is None or y is None:
            return False
        table_x, table_y = self.image_to_table(float(x), float(y))
        return -margin <= table_x <= 1.0 + margin and -margin <= table_y <= 1.0 + margin


def load_table_geometry(path: Path | None) -> TableGeometry | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return table_geometry_from_dict(payload)


def table_geometry_from_dict(payload: dict[str, Any]) -> TableGeometry:
    corners = payload.get("corners", payload)
    image_points = np.asarray([_corner_xy(corners[name]) for name in CORNER_NAMES], dtype=np.float64)
    table_points = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return TableGeometry(
        image_width=_optional_float(payload.get("image_width")),
        image_height=_optional_float(payload.get("image_height")),
        homography=_homography(image_points, table_points),
    )


def table_feature_names() -> list[str]:
    return [
        "table_x",
        "table_y",
        "table_inside",
        "table_outside_distance",
        "table_edge_margin",
        "table_center_distance",
        "table_net_distance",
    ]


def _corner_xy(value: Any) -> tuple[float, float]:
    if isinstance(value, dict):
        return float(value["x"]), float(value["y"])
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return float(value[0]), float(value[1])
    raise ValueError(f"Unsupported table corner value: {value!r}")


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _outside_unit_square_distance(x: float, y: float) -> float:
    dx = max(0.0 - x, 0.0, x - 1.0)
    dy = max(0.0 - y, 0.0, y - 1.0)
    return math.hypot(dx, dy)


def _homography(source: np.ndarray, destination: np.ndarray) -> np.ndarray:
    rows = []
    values = []
    for (x, y), (u, v) in zip(source, destination, strict=True):
        rows.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y])
        rows.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y])
        values.extend([u, v])
    solution = np.linalg.solve(np.asarray(rows, dtype=np.float64), np.asarray(values, dtype=np.float64))
    return np.asarray(
        [
            [solution[0], solution[1], solution[2]],
            [solution[3], solution[4], solution[5]],
            [solution[6], solution[7], 1.0],
        ],
        dtype=np.float64,
    )
