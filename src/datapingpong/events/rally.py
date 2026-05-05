from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from .trajectory import BallPoint


@dataclass(frozen=True)
class RallySegment:
    start_frame: int
    end_frame: int
    track_start_frame: int
    track_end_frame: int
    detected_points: int
    min_x: float | None
    max_x: float | None
    min_y: float | None
    max_y: float | None
    events: tuple[dict[str, Any], ...] = ()
    serve_like_start: bool = False
    serve_like_score: float | None = None
    pose_evidence_frames: int = 0
    toss_like_start: bool = False
    toss_rise_px: float | None = None
    toss_x_span_px: float | None = None
    bounce_x_direction_changes: int = 0
    max_bounce_x_delta: float | None = None
    start_nearest_wrist_distance: float | None = None
    end_nearest_wrist_distance: float | None = None
    rule_keep: bool = True
    rule_reject_reason: str | None = None

    @property
    def duration_frames(self) -> int:
        return self.end_frame - self.start_frame + 1


def detect_rallies(
    points: list[BallPoint],
    events: list[dict[str, Any]],
    pose_features: dict[int, dict[str, Any]] | None = None,
    *,
    max_ball_gap: int = 12,
    min_detected_points: int = 12,
    min_span_frames: int = 24,
    pre_padding: int = 12,
    post_padding: int = 12,
    merge_gap: int = 24,
    min_events_for_auto_keep: int = 1,
    no_event_max_detected_points: int = 96,
    no_event_max_span_frames: int = 120,
    low_event_max_events: int = 1,
    low_event_max_detected_points: int = 180,
    low_event_max_span_frames: int = 220,
    serve_lookback_frames: int = 36,
    serve_min_still_frames: int = 12,
    serve_min_score: float = 0.4,
    serve_toss_window_frames: int = 12,
    serve_min_toss_rise_px: float = 12.0,
    serve_max_toss_x_span_px: float = 24.0,
    serve_max_wrist_speed: float = 0.03,
    serve_max_center_speed: float = 0.006,
    serve_min_visible_ratio: float = 0.6,
    bounce_only_max_direction_changes: int = 0,
    bounce_only_max_abs_delta: float = 400.0,
    bounce_only_compact_max_span_x: float = 80.0,
    long_no_event_min_detected_points: int = 120,
    long_no_event_max_span_x: float = 140.0,
    serve_empty_max_detected_points: int = 60,
    serve_empty_max_span_x: float = 80.0,
    hit_only_max_span_x: float = 180.0,
    hit_only_max_span_frames: int = 220,
    continuation_gap: int = -1,
) -> list[RallySegment]:
    proposals = build_rally_proposals(
        points,
        events,
        pose_features,
        max_ball_gap=max_ball_gap,
        min_detected_points=min_detected_points,
        min_span_frames=min_span_frames,
        pre_padding=pre_padding,
        post_padding=post_padding,
        merge_gap=merge_gap,
        min_events_for_auto_keep=min_events_for_auto_keep,
        no_event_max_detected_points=no_event_max_detected_points,
        no_event_max_span_frames=no_event_max_span_frames,
        low_event_max_events=low_event_max_events,
        low_event_max_detected_points=low_event_max_detected_points,
        low_event_max_span_frames=low_event_max_span_frames,
        serve_lookback_frames=serve_lookback_frames,
        serve_min_still_frames=serve_min_still_frames,
        serve_min_score=serve_min_score,
        serve_toss_window_frames=serve_toss_window_frames,
        serve_min_toss_rise_px=serve_min_toss_rise_px,
        serve_max_toss_x_span_px=serve_max_toss_x_span_px,
        serve_max_wrist_speed=serve_max_wrist_speed,
        serve_max_center_speed=serve_max_center_speed,
        serve_min_visible_ratio=serve_min_visible_ratio,
        bounce_only_max_direction_changes=bounce_only_max_direction_changes,
        bounce_only_max_abs_delta=bounce_only_max_abs_delta,
        bounce_only_compact_max_span_x=bounce_only_compact_max_span_x,
        long_no_event_min_detected_points=long_no_event_min_detected_points,
        long_no_event_max_span_x=long_no_event_max_span_x,
        serve_empty_max_detected_points=serve_empty_max_detected_points,
        serve_empty_max_span_x=serve_empty_max_span_x,
        hit_only_max_span_x=hit_only_max_span_x,
        hit_only_max_span_frames=hit_only_max_span_frames,
    )
    kept = [proposal for proposal in proposals if proposal.rule_keep]
    return _merge_continued_rallies(kept, continuation_gap=continuation_gap)


def build_rally_proposals(
    points: list[BallPoint],
    events: list[dict[str, Any]],
    pose_features: dict[int, dict[str, Any]] | None = None,
    *,
    max_ball_gap: int = 12,
    min_detected_points: int = 12,
    min_span_frames: int = 24,
    pre_padding: int = 12,
    post_padding: int = 12,
    merge_gap: int = 24,
    min_events_for_auto_keep: int = 1,
    no_event_max_detected_points: int = 96,
    no_event_max_span_frames: int = 120,
    low_event_max_events: int = 1,
    low_event_max_detected_points: int = 180,
    low_event_max_span_frames: int = 220,
    serve_lookback_frames: int = 36,
    serve_min_still_frames: int = 12,
    serve_min_score: float = 0.4,
    serve_toss_window_frames: int = 12,
    serve_min_toss_rise_px: float = 12.0,
    serve_max_toss_x_span_px: float = 24.0,
    serve_max_wrist_speed: float = 0.03,
    serve_max_center_speed: float = 0.006,
    serve_min_visible_ratio: float = 0.6,
    bounce_only_max_direction_changes: int = 0,
    bounce_only_max_abs_delta: float = 400.0,
    bounce_only_compact_max_span_x: float = 80.0,
    long_no_event_min_detected_points: int = 120,
    long_no_event_max_span_x: float = 140.0,
    serve_empty_max_detected_points: int = 60,
    serve_empty_max_span_x: float = 80.0,
    hit_only_max_span_x: float = 180.0,
    hit_only_max_span_frames: int = 220,
) -> list[RallySegment]:
    detected_points = [
        point
        for point in sorted(points, key=lambda row: row.frame)
        if point.detected and point.x is not None and point.y is not None
    ]
    if not detected_points:
        return []

    raw_segments = _build_track_segments(detected_points, max_ball_gap=max_ball_gap)
    padded_segments = [
        RallySegment(
            start_frame=max(0, segment.start_frame - pre_padding),
            end_frame=segment.end_frame + post_padding,
            track_start_frame=segment.track_start_frame,
            track_end_frame=segment.track_end_frame,
            detected_points=segment.detected_points,
            min_x=segment.min_x,
            max_x=segment.max_x,
            min_y=segment.min_y,
            max_y=segment.max_y,
            serve_like_start=False,
            serve_like_score=None,
            pose_evidence_frames=0,
            toss_like_start=False,
            toss_rise_px=None,
            toss_x_span_px=None,
            bounce_x_direction_changes=0,
            max_bounce_x_delta=None,
            start_nearest_wrist_distance=None,
            end_nearest_wrist_distance=None,
            events=(),
        )
        for segment in raw_segments
    ]
    merged_segments = _merge_segments(padded_segments, merge_gap=merge_gap)
    sorted_events = sorted(events, key=lambda row: (int(row["frame"]), str(row["event"])))
    use_pose_gating = bool(pose_features)

    rallies: list[RallySegment] = []
    for segment in merged_segments:
        track_span_frames = segment.track_end_frame - segment.track_start_frame + 1
        rally_events = tuple(
            dict(row)
            for row in sorted_events
            if segment.start_frame <= int(row["frame"]) <= segment.end_frame
        )
        serve_like_start, serve_like_score, pose_evidence_frames = detect_serve_like_start(
            pose_features or {},
            track_start_frame=segment.track_start_frame,
            lookback_frames=serve_lookback_frames,
            min_still_frames=serve_min_still_frames,
            min_score=serve_min_score,
            max_wrist_speed=serve_max_wrist_speed,
            max_center_speed=serve_max_center_speed,
            min_visible_ratio=serve_min_visible_ratio,
        )
        toss_like_start, toss_rise_px, toss_x_span_px = detect_toss_like_start(
            detected_points,
            track_start_frame=segment.track_start_frame,
            window_frames=serve_toss_window_frames,
            min_rise_px=serve_min_toss_rise_px,
            max_x_span_px=serve_max_toss_x_span_px,
        )
        serve_like_start = serve_like_start and toss_like_start
        event_count = len(rally_events)
        bounce_count = sum(1 for event in rally_events if str(event["event"]) == "bounce")
        hit_count = sum(1 for event in rally_events if str(event["event"]) == "hit")
        span_x = _span(segment.min_x, segment.max_x)
        segment_points = _segment_points(detected_points, start_frame=segment.track_start_frame, end_frame=segment.track_end_frame)
        bounce_x_direction_changes, max_bounce_x_delta = bounce_direction_metrics(rally_events)
        start_nearest_wrist_distance = nearest_wrist_distance(pose_features or {}, point=segment_points[0] if segment_points else None)
        end_nearest_wrist_distance = nearest_wrist_distance(pose_features or {}, point=segment_points[-1] if segment_points else None)
        rule_keep = True
        rule_reject_reason = None
        if segment.detected_points < min_detected_points:
            rule_keep = False
            rule_reject_reason = "min_detected_points"
        elif track_span_frames < min_span_frames:
            rule_keep = False
            rule_reject_reason = "min_span_frames"
        if use_pose_gating:
            keep_empty_segment = (
                event_count >= min_events_for_auto_keep
                or segment.detected_points > no_event_max_detected_points
                or track_span_frames > no_event_max_span_frames
                or serve_like_start
            )
            if rule_keep and event_count == 0 and not keep_empty_segment:
                rule_keep = False
                rule_reject_reason = "no_event_non_serve"
            if (
                rule_keep
                and
                not serve_like_start
                and event_count <= low_event_max_events
                and segment.detected_points <= low_event_max_detected_points
                and track_span_frames <= low_event_max_span_frames
            ):
                rule_keep = False
                rule_reject_reason = "low_event_non_serve"
            if (
                rule_keep
                and not serve_like_start
                and hit_count == 0
                and bounce_count >= 2
                and (
                    (
                        bounce_x_direction_changes <= bounce_only_max_direction_changes
                        and max_bounce_x_delta is not None
                        and max_bounce_x_delta <= bounce_only_max_abs_delta
                    )
                    or (bounce_count >= 3 and span_x <= bounce_only_compact_max_span_x)
                )
            ):
                rule_keep = False
                rule_reject_reason = "bounce_only_handoff"
            if (
                rule_keep
                and serve_like_start
                and event_count == 0
                and segment.detected_points <= serve_empty_max_detected_points
                and span_x <= serve_empty_max_span_x
            ):
                rule_keep = False
                rule_reject_reason = "empty_serve_like_handoff"
            if (
                rule_keep
                and not serve_like_start
                and event_count == 0
                and segment.detected_points >= long_no_event_min_detected_points
                and span_x <= long_no_event_max_span_x
            ):
                rule_keep = False
                rule_reject_reason = "long_no_event_handoff"
            if (
                rule_keep
                and not serve_like_start
                and bounce_count == 0
                and hit_count >= 2
                and track_span_frames <= hit_only_max_span_frames
                and span_x <= hit_only_max_span_x
            ):
                rule_keep = False
                rule_reject_reason = "hit_only_handoff"
        rallies.append(
            RallySegment(
                start_frame=segment.start_frame,
                end_frame=segment.end_frame,
                track_start_frame=segment.track_start_frame,
                track_end_frame=segment.track_end_frame,
                detected_points=segment.detected_points,
                min_x=segment.min_x,
                max_x=segment.max_x,
                min_y=segment.min_y,
                max_y=segment.max_y,
                serve_like_start=serve_like_start,
                serve_like_score=serve_like_score,
                pose_evidence_frames=pose_evidence_frames,
                toss_like_start=toss_like_start,
                toss_rise_px=toss_rise_px,
                toss_x_span_px=toss_x_span_px,
                bounce_x_direction_changes=bounce_x_direction_changes,
                max_bounce_x_delta=max_bounce_x_delta,
                start_nearest_wrist_distance=start_nearest_wrist_distance,
                end_nearest_wrist_distance=end_nearest_wrist_distance,
                rule_keep=rule_keep,
                rule_reject_reason=rule_reject_reason,
                events=rally_events,
            )
        )
    return rallies


def rallies_to_dicts(rallies: list[RallySegment], *, fps: float) -> list[dict[str, Any]]:
    rows = []
    for index, rally in enumerate(rallies, start=1):
        event_counts: dict[str, int] = {}
        bounce_positions = []
        hit_positions = []
        for event in rally.events:
            event_name = str(event["event"])
            event_counts[event_name] = event_counts.get(event_name, 0) + 1
            event_row = {
                "frame": int(event["frame"]),
                "x": event.get("x"),
                "y": event.get("y"),
                "probability": event.get("probability"),
            }
            if event_name == "bounce":
                bounce_positions.append(event_row)
            if event_name == "hit":
                hit_positions.append(event_row)

        rows.append(
            {
                "id": index,
                "start_frame": rally.start_frame,
                "end_frame": rally.end_frame,
                "duration_frames": rally.duration_frames,
                "duration_s": round(rally.duration_frames / fps, 3) if fps > 0 else None,
                "track_start_frame": rally.track_start_frame,
                "track_end_frame": rally.track_end_frame,
                "detected_points": rally.detected_points,
                "event_count": len(rally.events),
                "event_counts": event_counts,
                "first_event_frame": min((int(event["frame"]) for event in rally.events), default=None),
                "last_event_frame": max((int(event["frame"]) for event in rally.events), default=None),
                "bounds": {
                    "min_x": rally.min_x,
                    "max_x": rally.max_x,
                    "min_y": rally.min_y,
                    "max_y": rally.max_y,
                },
                "serve_like_start": rally.serve_like_start,
                "serve_like_score": rally.serve_like_score,
                "pose_evidence_frames": rally.pose_evidence_frames,
                "toss_like_start": rally.toss_like_start,
                "toss_rise_px": rally.toss_rise_px,
                "toss_x_span_px": rally.toss_x_span_px,
                "rule_keep": rally.rule_keep,
                "rule_reject_reason": rally.rule_reject_reason,
                "bounce_positions": bounce_positions,
                "hit_positions": hit_positions,
                "events": list(rally.events),
            }
        )
    return rows


def _build_track_segments(points: list[BallPoint], *, max_ball_gap: int) -> list[RallySegment]:
    segments: list[RallySegment] = []
    current: list[BallPoint] = []
    previous_frame: int | None = None
    for point in points:
        if previous_frame is not None and point.frame - previous_frame > max_ball_gap + 1:
            segments.append(_segment_from_points(current))
            current = []
        current.append(point)
        previous_frame = point.frame
    if current:
        segments.append(_segment_from_points(current))
    return segments


def _segment_from_points(points: list[BallPoint]) -> RallySegment:
    xs = [float(point.x) for point in points if point.x is not None]
    ys = [float(point.y) for point in points if point.y is not None]
    return RallySegment(
        start_frame=points[0].frame,
        end_frame=points[-1].frame,
        track_start_frame=points[0].frame,
        track_end_frame=points[-1].frame,
        detected_points=len(points),
        min_x=min(xs) if xs else None,
        max_x=max(xs) if xs else None,
        min_y=min(ys) if ys else None,
        max_y=max(ys) if ys else None,
        serve_like_start=False,
        serve_like_score=None,
        pose_evidence_frames=0,
        toss_like_start=False,
        toss_rise_px=None,
        toss_x_span_px=None,
        rule_keep=True,
        rule_reject_reason=None,
        events=(),
    )


def _merge_segments(segments: list[RallySegment], *, merge_gap: int) -> list[RallySegment]:
    if not segments:
        return []
    ordered = sorted(segments, key=lambda row: row.start_frame)
    merged = [ordered[0]]
    for segment in ordered[1:]:
        previous = merged[-1]
        if segment.start_frame - previous.end_frame > merge_gap:
            merged.append(segment)
            continue
        merged[-1] = RallySegment(
            start_frame=min(previous.start_frame, segment.start_frame),
            end_frame=max(previous.end_frame, segment.end_frame),
            track_start_frame=min(previous.track_start_frame, segment.track_start_frame),
            track_end_frame=max(previous.track_end_frame, segment.track_end_frame),
            detected_points=previous.detected_points + segment.detected_points,
            min_x=_combine_min(previous.min_x, segment.min_x),
            max_x=_combine_max(previous.max_x, segment.max_x),
            min_y=_combine_min(previous.min_y, segment.min_y),
            max_y=_combine_max(previous.max_y, segment.max_y),
            serve_like_start=previous.serve_like_start or segment.serve_like_start,
            serve_like_score=_combine_max(previous.serve_like_score, segment.serve_like_score),
                pose_evidence_frames=max(previous.pose_evidence_frames, segment.pose_evidence_frames),
                toss_like_start=previous.toss_like_start or segment.toss_like_start,
                toss_rise_px=_combine_max(previous.toss_rise_px, segment.toss_rise_px),
                toss_x_span_px=_combine_max(previous.toss_x_span_px, segment.toss_x_span_px),
                bounce_x_direction_changes=0,
                max_bounce_x_delta=None,
                start_nearest_wrist_distance=None,
                end_nearest_wrist_distance=None,
                rule_keep=previous.rule_keep and segment.rule_keep,
                rule_reject_reason=previous.rule_reject_reason or segment.rule_reject_reason,
                events=(),
        )
    return merged


def _merge_continued_rallies(rallies: list[RallySegment], *, continuation_gap: int) -> list[RallySegment]:
    if continuation_gap < 0 or not rallies:
        return rallies
    ordered = sorted(rallies, key=lambda row: row.start_frame)
    merged = [ordered[0]]
    for rally in ordered[1:]:
        previous = merged[-1]
        gap = rally.start_frame - previous.end_frame - 1
        if gap > continuation_gap or rally.serve_like_start:
            merged.append(rally)
            continue
        merged_events = tuple(sorted((*previous.events, *rally.events), key=lambda row: (int(row["frame"]), str(row["event"]))))
        merged_bounce_changes, merged_max_bounce_delta = bounce_direction_metrics(merged_events)
        merged[-1] = RallySegment(
            start_frame=previous.start_frame,
            end_frame=rally.end_frame,
            track_start_frame=previous.track_start_frame,
            track_end_frame=rally.track_end_frame,
            detected_points=previous.detected_points + rally.detected_points,
            min_x=_combine_min(previous.min_x, rally.min_x),
            max_x=_combine_max(previous.max_x, rally.max_x),
            min_y=_combine_min(previous.min_y, rally.min_y),
            max_y=_combine_max(previous.max_y, rally.max_y),
            events=merged_events,
            serve_like_start=previous.serve_like_start,
            serve_like_score=previous.serve_like_score,
            pose_evidence_frames=max(previous.pose_evidence_frames, rally.pose_evidence_frames),
            toss_like_start=previous.toss_like_start,
            toss_rise_px=previous.toss_rise_px,
            toss_x_span_px=previous.toss_x_span_px,
            bounce_x_direction_changes=merged_bounce_changes,
            max_bounce_x_delta=merged_max_bounce_delta,
            start_nearest_wrist_distance=previous.start_nearest_wrist_distance,
            end_nearest_wrist_distance=rally.end_nearest_wrist_distance,
            rule_keep=True,
            rule_reject_reason=None,
        )
    return merged


def _combine_min(left: float | None, right: float | None) -> float | None:
    values = [value for value in (left, right) if value is not None]
    return min(values) if values else None


def _combine_max(left: float | None, right: float | None) -> float | None:
    values = [value for value in (left, right) if value is not None]
    return max(values) if values else None


def _span(min_value: float | None, max_value: float | None) -> float:
    if min_value is None or max_value is None:
        return 0.0
    return float(max_value - min_value)


def _segment_points(points: list[BallPoint], *, start_frame: int, end_frame: int) -> list[BallPoint]:
    return [
        point
        for point in points
        if point.detected and point.x is not None and point.y is not None and start_frame <= point.frame <= end_frame
    ]


def bounce_direction_metrics(events: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> tuple[int, float | None]:
    bounce_x = [float(event["x"]) for event in events if str(event["event"]) == "bounce" and event.get("x") is not None]
    if len(bounce_x) < 2:
        return 0, None
    deltas = [right - left for left, right in zip(bounce_x, bounce_x[1:])]
    signs = [1 if delta > 0 else -1 for delta in deltas if abs(delta) >= 1e-6]
    direction_changes = sum(1 for left, right in zip(signs, signs[1:]) if left != right)
    return direction_changes, max(abs(delta) for delta in deltas)


def nearest_wrist_distance(
    pose_features: dict[int, dict[str, Any]],
    *,
    point: BallPoint | None,
) -> float | None:
    if point is None or point.x is None or point.y is None:
        return None
    row = pose_features.get(point.frame)
    if row is None:
        return None
    ball_x = float(point.x) / 1920.0
    ball_y = float(point.y) / 1080.0
    distances: list[float] = []
    for player in row.get("players", {}).values():
        if float(player.get("pose_detected", 0.0)) < 0.5:
            continue
        body_scale = max(
            float(player.get("shoulder_width", 0.0)),
            float(player.get("torso_length", 0.0)),
            float(player.get("hip_width", 0.0)),
            1e-6,
        )
        for side in ("left", "right"):
            wrist_x = float(player.get("shoulder_center_x", 0.0)) + float(player.get(f"{side}_wrist_rel_x", 0.0)) * body_scale
            wrist_y = float(player.get("shoulder_center_y", 0.0)) + float(player.get(f"{side}_wrist_rel_y", 0.0)) * body_scale
            distances.append(math.hypot(ball_x - wrist_x, ball_y - wrist_y))
    return round(min(distances), 6) if distances else None


def detect_serve_like_start(
    pose_features: dict[int, dict[str, Any]],
    *,
    track_start_frame: int,
    lookback_frames: int,
    min_still_frames: int,
    min_score: float,
    max_wrist_speed: float,
    max_center_speed: float,
    min_visible_ratio: float,
) -> tuple[bool, float | None, int]:
    if not pose_features:
        return False, None, 0
    start = max(0, track_start_frame - lookback_frames)
    frames = [frame for frame in range(start, track_start_frame) if frame in pose_features]
    if not frames:
        return False, None, 0

    longest_still = 0
    current_still = 0
    motion_scores: list[float] = []
    evidence_frames = 0
    previous_rows: dict[str, dict[str, Any]] = {}
    for frame in frames:
        row = pose_features[frame]
        players = row.get("players", {})
        role_results: list[tuple[bool, float]] = []
        role_visible: list[bool] = []
        for role in ("left", "right"):
            player = players.get(role, {})
            previous = previous_rows.get(role)
            is_valid, motion_score = _player_stillness(
                player,
                previous,
                max_wrist_speed=max_wrist_speed,
                max_center_speed=max_center_speed,
                min_visible_ratio=min_visible_ratio,
            )
            role_visible.append(_player_pose_available(player, min_visible_ratio=min_visible_ratio))
            role_results.append((is_valid, motion_score))
            previous_rows[role] = player

        if all(role_visible):
            evidence_frames += 1
        if all(result[0] for result in role_results):
            current_still += 1
            longest_still = max(longest_still, current_still)
            motion_scores.append(max(result[1] for result in role_results))
        else:
            current_still = 0

    score = None
    if motion_scores:
        score = round(max(0.0, 1.0 - (sum(motion_scores) / len(motion_scores))), 6)
    return longest_still >= min_still_frames and score is not None and score >= min_score, score, evidence_frames


def detect_toss_like_start(
    points: list[BallPoint],
    *,
    track_start_frame: int,
    window_frames: int,
    min_rise_px: float,
    max_x_span_px: float,
) -> tuple[bool, float | None, float | None]:
    window_points = [
        point
        for point in points
        if point.detected and point.x is not None and point.y is not None and track_start_frame <= point.frame < track_start_frame + window_frames
    ]
    if len(window_points) < 3:
        return False, None, None
    first_y = float(window_points[0].y)
    rise_px = round(first_y - min(float(point.y) for point in window_points), 3)
    x_span_px = round(max(float(point.x) for point in window_points) - min(float(point.x) for point in window_points), 3)
    return rise_px >= min_rise_px and x_span_px <= max_x_span_px, rise_px, x_span_px


def _player_stillness(
    player: dict[str, Any],
    previous: dict[str, Any] | None,
    *,
    max_wrist_speed: float,
    max_center_speed: float,
    min_visible_ratio: float,
) -> tuple[bool, float]:
    if not _player_pose_available(player, min_visible_ratio=min_visible_ratio):
        return False, 1.0

    wrist_speed = max(
        math.hypot(float(player.get("left_wrist_velocity_x", 0.0)), float(player.get("left_wrist_velocity_y", 0.0))),
        math.hypot(float(player.get("right_wrist_velocity_x", 0.0)), float(player.get("right_wrist_velocity_y", 0.0))),
    )
    center_speed = 0.0
    if previous is not None and float(previous.get("pose_detected", 0.0)) >= 0.5:
        center_speed = max(
            math.hypot(
                float(player.get("shoulder_center_x", 0.0)) - float(previous.get("shoulder_center_x", 0.0)),
                float(player.get("shoulder_center_y", 0.0)) - float(previous.get("shoulder_center_y", 0.0)),
            ),
            math.hypot(
                float(player.get("hip_center_x", 0.0)) - float(previous.get("hip_center_x", 0.0)),
                float(player.get("hip_center_y", 0.0)) - float(previous.get("hip_center_y", 0.0)),
            ),
        )

    wrist_ratio = wrist_speed / max(max_wrist_speed, 1e-6)
    center_ratio = center_speed / max(max_center_speed, 1e-6)
    motion_score = max(wrist_ratio, center_ratio)
    return wrist_speed <= max_wrist_speed and center_speed <= max_center_speed, motion_score


def _player_pose_available(player: dict[str, Any], *, min_visible_ratio: float) -> bool:
    return float(player.get("pose_detected", 0.0)) >= 0.5 and float(player.get("upper_body_visible_ratio", 0.0)) >= min_visible_ratio
