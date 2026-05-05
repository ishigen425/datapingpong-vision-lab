from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

from .trajectory import BallPoint


@dataclass(frozen=True)
class HitStroke:
    event_frame: int
    ball_frame: int | None
    player_role: str | None
    striking_arm: str | None
    contact_side: str | None
    stroke_side: str | None
    confidence: float | None
    ball_wrist_distance: float | None
    wrist_speed: float | None


def infer_hit_strokes(
    events: list[dict[str, Any]],
    points: list[BallPoint],
    pose_features: dict[int, dict[str, Any]],
    *,
    frame_width: int = 1920,
    frame_height: int = 1080,
    window_frames: int = 4,
    left_racket_hand: str = "right",
    right_racket_hand: str = "right",
) -> list[HitStroke]:
    point_by_frame = {point.frame: point for point in points if point.detected and point.x is not None and point.y is not None}
    strokes: list[HitStroke] = []
    for event in events:
        if str(event["event"]) != "hit":
            continue
        event_frame = int(event["frame"])
        candidate = best_hit_candidate(
            event_frame=event_frame,
            point_by_frame=point_by_frame,
            pose_features=pose_features,
            frame_width=frame_width,
            frame_height=frame_height,
            window_frames=window_frames,
        )
        if candidate is None:
            strokes.append(
                HitStroke(
                    event_frame=event_frame,
                    ball_frame=None,
                    player_role=None,
                    striking_arm=None,
                    contact_side=None,
                    stroke_side=None,
                    confidence=None,
                    ball_wrist_distance=None,
                    wrist_speed=None,
                )
            )
            continue
        racket_hand = left_racket_hand if candidate["role"] == "left" else right_racket_hand
        contact_side = infer_contact_side(candidate["ball_x"], candidate["shoulder_x"])
        stroke_side = infer_stroke_side(candidate["role"], contact_side, racket_hand=racket_hand)
        strokes.append(
            HitStroke(
                event_frame=event_frame,
                ball_frame=int(candidate["ball_frame"]),
                player_role=str(candidate["role"]),
                striking_arm=str(candidate["arm"]),
                contact_side=contact_side,
                stroke_side=stroke_side,
                confidence=hit_confidence(candidate["distance"], candidate.get("second_distance")),
                ball_wrist_distance=round(float(candidate["distance"]), 6),
                wrist_speed=round(float(candidate["wrist_speed"]), 6),
            )
        )
    return strokes


def strokes_to_dicts(rows: list[HitStroke]) -> list[dict[str, Any]]:
    return [asdict(row) for row in rows]


def best_hit_candidate(
    *,
    event_frame: int,
    point_by_frame: dict[int, BallPoint],
    pose_features: dict[int, dict[str, Any]],
    frame_width: int,
    frame_height: int,
    window_frames: int,
) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for frame in range(event_frame - window_frames, event_frame + window_frames + 1):
        point = point_by_frame.get(frame)
        row = pose_features.get(frame)
        if point is None or row is None:
            continue
        ball_x = float(point.x) / max(1.0, float(frame_width))
        ball_y = float(point.y) / max(1.0, float(frame_height))
        for role, player in row.get("players", {}).items():
            if float(player.get("pose_detected", 0.0)) < 0.5:
                continue
            for arm in ("left", "right"):
                if float(player.get(f"{arm}_arm_visible", 0.0)) < 0.5:
                    continue
                wrist = wrist_position(player, arm)
                if wrist is None:
                    continue
                distance = math.hypot(ball_x - wrist[0], ball_y - wrist[1])
                candidates.append(
                    {
                        "frame_gap": abs(frame - event_frame),
                        "ball_frame": frame,
                        "role": role,
                        "arm": arm,
                        "distance": distance,
                        "wrist_speed": wrist_speed(player, arm),
                        "ball_x": ball_x,
                        "shoulder_x": float(player.get("shoulder_center_x", 0.0)),
                    }
                )
    if not candidates:
        return None
    ordered = sorted(candidates, key=lambda row: (row["distance"], -row["wrist_speed"], row["frame_gap"]))
    best = dict(ordered[0])
    best["second_distance"] = ordered[1]["distance"] if len(ordered) > 1 else None
    return best


def wrist_position(player: dict[str, Any], arm: str) -> tuple[float, float] | None:
    if float(player.get(f"{arm}_arm_visible", 0.0)) < 0.5:
        return None
    body_scale = max(
        float(player.get("shoulder_width", 0.0)),
        float(player.get("torso_length", 0.0)),
        float(player.get("hip_width", 0.0)),
        1e-6,
    )
    return (
        float(player.get("shoulder_center_x", 0.0)) + float(player.get(f"{arm}_wrist_rel_x", 0.0)) * body_scale,
        float(player.get("shoulder_center_y", 0.0)) + float(player.get(f"{arm}_wrist_rel_y", 0.0)) * body_scale,
    )


def wrist_speed(player: dict[str, Any], arm: str) -> float:
    return math.hypot(
        float(player.get(f"{arm}_wrist_velocity_x", 0.0)),
        float(player.get(f"{arm}_wrist_velocity_y", 0.0)),
    )


def infer_contact_side(ball_x: float, shoulder_x: float) -> str:
    return "left" if ball_x < shoulder_x else "right"


def infer_stroke_side(role: str, contact_side: str, *, racket_hand: str) -> str:
    if role not in {"left", "right"}:
        return "unknown"
    if racket_hand not in {"left", "right"}:
        return "unknown"
    forehand_side = "right" if role == "left" else "left"
    if racket_hand == "left":
        forehand_side = "left" if forehand_side == "right" else "right"
    return "forehand" if contact_side == forehand_side else "backhand"


def hit_confidence(best_distance: float, second_distance: float | None) -> float | None:
    base = max(0.0, 1.0 - best_distance / 0.6)
    if second_distance is None:
        return round(base, 6)
    margin = max(0.0, min(1.0, (second_distance - best_distance) / 0.4))
    return round((base + margin) / 2.0, 6)
