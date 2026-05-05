# detect_rallies

Segments a match into rally frame ranges from ball visibility spans, bounce/hit events, and optional pose-based serve-start cues.

```bash
docker compose run --rm app python tasks/detect_rallies/run.py
```

The task is meant to create the rally-level container needed for later stroke collection and scoring. It uses:

- ball detection continuity (`--max-ball-gap`)
- minimum visible-ball support (`--min-detected-points`, `--min-span-frames`)
- small padding before and after each visible-ball segment
- segment merging when brief gaps likely belong to the same rally
- optional `left/right` pose stillness near the segment start to distinguish serve-like starts from handoff-like ball exchanges
- conservative pruning of short non-serve segments with no events or only a single weak event

For the default local DJI legacy ball JSON, the task applies the known `+4` frame shift by default:

```bash
docker compose run --rm app python tasks/detect_rallies/run.py \
  --ball data/annotations/ball_tracking/DJI_0056_001_predictions.json \
  --ball-frame-offset 4
```

Use belief-tracker outputs instead:

```bash
docker compose run --rm app python tasks/detect_rallies/run.py \
  --ball outputs/detect_ball_legacy/DJI_0056_001_belief_predictions.json \
  --ball-frame-offset 0 \
  --events outputs/detect_events_from_ball/DJI_0056_001_belief_events.json
```

The pose-assisted defaults now try to suppress short no-event ball exchanges unless they look like a real serve start. The rule is intentionally simple: if both players are visible and comparatively still just before the ball segment begins, the segment is more likely to be a serve-start rally than a between-points handoff.

`--serve-min-score` sets how strong that stillness evidence must be. Raising it makes serve detection stricter and usually reduces false rally starts from handoff segments.

The current defaults also require a small toss pattern near the start: the ball should rise upward within the first few frames and should not drift too far sideways while doing so. Tune this with `--serve-min-toss-rise-px` and `--serve-max-toss-x-span-px`.

There is now a bounded handoff-suppression pass for a few common false positives:

- monotonic bounce-only segments with no hit
- empty serve-like/toss-like starts
- long no-event passes with limited x-span
- short hit-only segments with no bounce

Proposal diagnostics now also include bounce-direction-change counts and nearest-wrist distance at the start/end of each segment.

If you want to experiment with merging consecutive kept segments after rule filtering, you can enable an optional final post-pass with `--continuation-gap`. It is disabled by default because a naive global merge tends to over-merge the DJI sample.

The output JSON includes one row per rally with:

- `start_frame` / `end_frame`
- `track_start_frame` / `track_end_frame`
- `duration_frames` / `duration_s`
- `event_counts`
- `serve_like_start`
- `serve_like_score`
- `bounce_positions`
- `hit_positions`

This is intentionally a first pass: it finds rally ranges, but it does not yet infer serve legality, point winner, or score.
