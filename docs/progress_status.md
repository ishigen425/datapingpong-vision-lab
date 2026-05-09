# Progress Status

This document summarizes the current state of the video-to-rally-metadata pipeline and the next improvement priorities.

## Goal

The project is now able to run an end-to-end experimental workflow from a table-tennis video toward rally metadata:

1. detect or load ball coordinates
2. detect bounce and hit events from the trajectory
3. segment rally ranges
4. estimate player pose
5. extract pose features
6. extract per-hit stroke details
7. render review videos and evaluate selected outputs

The implementation is useful for experimentation and review, but the current accuracy is not yet practical enough for reliable downstream analysis.

## Current Pipeline State

| Area | Current state | Main risk |
| --- | --- | --- |
| Ball tracking | Legacy UNet detector and imported local coordinates are available. The detector now has argmax, top-k track, and belief decoders. | Tracking gaps, jitter, false ball detections, and frame alignment errors propagate into every later step. |
| Bounce detection | Coordinate rules with table geometry are working. Local DJI best recorded F1 is `0.749`. | Remaining false positives/false negatives need table-relative and trajectory-quality analysis. |
| Hit detection | Horizontal velocity reversal plus table-prior and local-motion gates are working. Local DJI best recorded F1 is `0.550`. | Ball-only hit detection confuses racket contact with tracking noise, handoffs, non-rally motion, and ball reappearance after misses. |
| Rally detection | Rally ranges can be proposed from ball continuity, events, serve-like pose cues, toss cues, and handoff-suppression rules. | Start/end boundaries and false rally starts still need manual proposal labels and evaluation. |
| Pose estimation | MediaPipe Pose output and body-relative feature extraction are available. | Pose can be unstable under occlusion, distance, fast arm movement, and player-side ambiguity. |
| Stroke details | Per-hit fields such as player role, striking arm, contact side, stroke side, confidence, ball-wrist distance, and wrist speed are exported. | Output quality is limited by hit accuracy and pose quality; low-confidence or unknown cases need explicit handling. |
| Event diagnostics | `tasks/diagnose_event_detection` exports TP/FP/FN review tables with event scores, ball context, and hit local-motion diagnostics. | The task identifies failure candidates, but human video review is still needed to assign root causes. |
| Bounce ML detector | OpenTTGames softmax trained with `none/bounce/net_hit` can emit/evaluate `bounce` only. Current OpenTTGames game/test bounce F1 is `0.944`. | Local transfer still needs calibration and table-geometry handling. |
| Time-series review | `tasks/plot_ball_timeseries` exports SVG plots of image `x/y` and table-relative `table_x/table_y`, with predicted/reference event markers and diagnostic windows. | Human review should categorize whether errors are tracking, event scoring, table-position, or annotation-tolerance issues. |
| Review videos | Event, bounce, and multimodal overlays are available. | Video review is still useful after time-series review narrows the error set. |

## Baseline Metrics

The most relevant local DJI event-detection baseline is:

```bash
docker compose run --rm app python tasks/detect_events_from_ball/run.py \
  --input data/annotations/ball_tracking/DJI_0056_001_predictions.json \
  --output outputs/detect_events_from_ball/local_table_prior_motion_x30_y10_events.json \
  --table-geometry data/annotations/table_geometry/DJI_0056_001.json \
  --bounce-threshold 0.35 \
  --hit-threshold 0.55 \
  --smooth-window 3 \
  --hit-smooth-window 7 \
  --nms-window 3 \
  --arbitration-window 4 \
  --hit-local-window 6 \
  --hit-min-local-x-span 30 \
  --hit-min-local-y-span 10 \
  --hit-min-local-detections 4
```

Evaluation against `data/annotations/events/bounce_and_hit_frames.json` with `--tolerance 4`:

| Event | Precision | Recall | F1 | Predicted |
| --- | ---: | ---: | ---: | ---: |
| bounce | 0.796 | 0.708 | 0.749 | 216 |
| hit | 0.550 | 0.550 | 0.550 | 249 |

The current higher-precision hit setting adds `--hit-reject-same-directional-x-displacement 30` with `x35/y15` local motion gates:

| Event | Precision | Recall | F1 | Predicted |
| --- | ---: | ---: | ---: | ---: |
| bounce | 0.793 | 0.708 | 0.748 | 217 |
| hit | 0.641 | 0.510 | 0.568 | 198 |

Important caveats:

- OpenTTGames `net_hit` is not equivalent to the local DJI `hit` label.
- OpenTTGames softmax results should not be interpreted as local racket-hit performance.
- The `x40/y10` local-motion hit gate improved precision but looked too sparse visually.
- `--hit-min-directional-x-displacement 8` was too strict on the local sample and reduced hit F1 to `0.341`.

See `docs/event_detection_experiments.md` for the detailed experiment log.

## Current Diagnostic Output

Detailed TP/FP/FN rows can be generated with:

```bash
docker compose run --rm app python tasks/diagnose_event_detection/run.py \
  --predictions outputs/detect_events_from_ball/local_table_prior_motion_x30_y10_events.json
```

Current generated outputs:

```text
outputs/diagnose_event_detection/summary.json
outputs/diagnose_event_detection/details.json
outputs/diagnose_event_detection/details.csv
```

Time-series SVG review can be generated with:

```bash
docker compose run --rm app python tasks/plot_ball_timeseries/run.py \
  --events outputs/detect_events_from_ball/local_table_prior_motion_x30_y10_events.json \
  --reference data/annotations/events/bounce_and_hit_frames.json \
  --table-geometry data/annotations/table_geometry/DJI_0056_001.json
```

Diagnostic FP/FN windows can be generated with:

```bash
docker compose run --rm app python tasks/plot_ball_timeseries/run.py \
  --events outputs/detect_events_from_ball/local_table_prior_motion_x30_y10_events.json \
  --reference data/annotations/events/bounce_and_hit_frames.json \
  --table-geometry data/annotations/table_geometry/DJI_0056_001.json \
  --diagnostics outputs/diagnose_event_detection/details.csv \
  --diagnostic-status false_positive false_negative \
  --window-radius 45
```

The first diagnostic run confirms the same event metrics as the baseline and adds useful review cues:

| Event | FP mean probability | FN mean nearest prediction delta | FN with detected ball context |
| --- | ---: | ---: | ---: |
| bounce | 0.512 | 55.55 frames | 57 / 71 |
| hit | 0.709 | 29.78 frames | 92 / 112 |

Interpretation:

- Many hit false positives still have high rule probability, so threshold-only tuning is unlikely to be enough.
- Many hit false negatives have nearby ball context, so pose/racket context and better event arbitration are likely useful after tracking diagnostics.
- Bounce false negatives should be reviewed by table position and ball-track quality before adding a larger model.
- The same-direction hit rejection confirms that local x/y trajectory shape can improve precision without changing the ball tracker.

## Improvement Order

The next work should focus on upstream accuracy first, because later metadata quality depends on it.

1. Use the OpenTTGames-trained bounce detector as the primary learned bounce baseline.
2. Calibrate bounce prediction on the local DJI sample with table geometry and threshold review.
3. Improve ball tracking and tracking-quality diagnostics where bounce misses are tracking-related.
4. Defer hit ML until manually verified hit labels exist.
5. Try bounce-only rally extraction before adding hit labels.
6. Add targeted review clips for remaining false positives and false negatives.

## Ball Tracking Improvement Plan

Primary objective: make downstream event detection aware of coordinate quality instead of treating every ball point as equally reliable.

Planned improvements:

- Compare legacy imported coordinates and `detect_ball_legacy` belief-decoder outputs over the same frame ranges.
- Classify tracking failures into gaps, jitter, false ball detections, impossible jumps, occlusion, and frame-offset issues.
- Add or export per-frame tracking diagnostics such as detection confidence, local continuity, speed, acceleration, gap length, and interpolation status.
- Add trajectory cleanup for physically implausible jumps and short isolated detections.
- Use table/scene ROI to reject obvious off-table or background false positives where appropriate.

Human checks needed:

- Review whether hit and bounce misses are caused by event logic or upstream ball-coordinate failures.
- Inspect whether the known local DJI frame offset is correctly handled for each input source.
- Identify common false-ball sources in the local video.

## Bounce And Hit Improvement Plan

Primary objective: raise bounce/hit quality on the local DJI sample before relying on rally and stroke metadata.

Bounce improvements:

- Keep table geometry as a default local prior.
- Analyze false positives and false negatives by table-relative position.
- Sweep class-specific thresholds and NMS windows.
- Separate true missed bounces from annotation/frame-tolerance disagreements.
- Consider table-relative features for a lightweight local classifier once enough labeled local examples are available.

Hit improvements:

- Use current diagnostic fields to group false positives and false negatives:
  - `local_x_span`
  - `local_y_span`
  - `local_detections`
  - `local_before_x_displacement`
  - `local_after_x_displacement`
- Separate errors caused by ball jitter, non-rally handoffs, bounce/hit arbitration, and missed racket contact.
- Add pose features around hit candidates, especially nearest wrist distance, wrist speed, side/player consistency, and pose confidence.
- Treat low-evidence hit candidates as lower confidence instead of forcing a binary decision.
- Preserve high-recall candidate generation, then use stricter classifier or confidence filtering downstream.

Human checks needed:

- Review hit false positives and decide whether they are genuinely wrong, label mismatches, or ambiguous contacts.
- Review hit false negatives and decide whether the ball track, pose, or event rule failed.
- Confirm whether the `+-4` frame tolerance is appropriate for practical metadata use.

## Rally And Stroke Follow-Up

Rally detection should wait until ball and contact events are less noisy, but the next useful rally-specific step is labeling proposal rows from:

```text
outputs/detect_rallies/DJI_0056_001_rally_proposals.json
```

Suggested labels:

- `true_rally`
- `handoff`
- `serve_miss_or_ace`
- `warmup_or_between_points`
- `tracking_noise`
- `unclear`

After labeling, run:

```bash
docker compose run --rm app python tasks/evaluate_rally_proposals/run.py \
  --proposals outputs/detect_rallies/DJI_0056_001_rally_proposals.json
```

Stroke details should remain confidence-oriented until hit timing is stronger. `unknown` should be a valid output when pose or contact evidence is weak.

## Practical Targets

Near-term targets for the local DJI sample:

- bounce F1: `0.80+`
- hit F1: `0.70` range after adding pose/racket context
- rally proposal precision: `0.85+` for high-confidence accepted rallies
- stroke details: reliable output on a high-confidence subset before trying to classify every hit

The practical path is to extract high-confidence rally metadata first, then expand coverage as diagnostics and labels improve.
