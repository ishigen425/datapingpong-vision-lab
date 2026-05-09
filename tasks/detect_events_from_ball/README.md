# detect_events_from_ball

Scores bounce and hit probabilities from ball coordinates using trajectory features only.

```bash
docker compose run --rm app python tasks/detect_events_from_ball/run.py
```

Inputs can be the legacy ball JSON, detector prediction JSON, OpenTTGames frame-to-coordinate JSON, or normalized JSONL from `tasks/import_openttgames`.
Pass `--table-geometry data/annotations/table_geometry/<video>.json` to keep bounce candidates on the annotated table region.

The imported local DJI legacy ball JSON is about 4 frames early relative to the video because the original detector used a 9-frame window. For that file, pass `--input-frame-offset 4` to shift coordinates onto the center frame before event scoring.

Default thresholds are tuned for the imported local `DJI_0056_001` annotation:

- bounce: `0.35`
- hit: `0.65`
- NMS window: `8` frames

For the local DJI sample, horizontal velocity reversal is now a primary hit cue. A more recall-oriented hit run is:

```bash
docker compose run --rm app python tasks/detect_events_from_ball/run.py \
  --input data/annotations/ball_tracking/DJI_0056_001_predictions.json \
  --output outputs/detect_events_from_ball/local_table_xflip_hit_events.json \
  --table-geometry data/annotations/table_geometry/DJI_0056_001.json \
  --bounce-threshold 0.35 \
  --hit-threshold 0.45 \
  --smooth-window 3 \
  --nms-window 3 \
  --hit-suppression-window 0
```

The current table-prior hit/bounce run uses separate smoothing for bounce and hit, then resolves close hit/bounce conflicts by table zone:

```bash
docker compose run --rm app python tasks/detect_events_from_ball/run.py \
  --input data/annotations/ball_tracking/DJI_0056_001_predictions.json \
  --output outputs/detect_events_from_ball/local_table_prior_events.json \
  --table-geometry data/annotations/table_geometry/DJI_0056_001.json \
  --bounce-threshold 0.35 \
  --hit-threshold 0.55 \
  --smooth-window 3 \
  --hit-smooth-window 7 \
  --nms-window 3 \
  --arbitration-window 4
```

If hit labels are too dense in non-rally or poor-tracking sections, require local ball motion around hit candidates. This keeps x-direction reversal as the main hit cue, but rejects short jitter where the ball barely moves. The current balanced local DJI setting is `x30/y10`; `x40/y10` is higher-precision but too sparse by visual inspection.

```bash
docker compose run --rm app python tasks/detect_events_from_ball/run.py \
  --input data/annotations/ball_tracking/DJI_0056_001_predictions.json \
  --input-frame-offset 4 \
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

The detector also supports `--hit-min-directional-x-displacement` for stricter pre/post x-direction reversal checks. On the current DJI sample, `8px` was too strict and dropped hit recall sharply, so it is not part of the recommended run.

For a higher-precision local DJI hit setting, reject candidates that continue strongly in the same x direction before and after the candidate frame:

```bash
docker compose run --rm app python tasks/detect_events_from_ball/run.py \
  --input data/annotations/ball_tracking/DJI_0056_001_predictions.json \
  --output outputs/detect_events_from_ball/local_table_prior_motion_x35_y15_same30_events.json \
  --table-geometry data/annotations/table_geometry/DJI_0056_001.json \
  --bounce-threshold 0.35 \
  --hit-threshold 0.55 \
  --smooth-window 3 \
  --hit-smooth-window 7 \
  --nms-window 3 \
  --arbitration-window 4 \
  --hit-local-window 6 \
  --hit-min-local-x-span 35 \
  --hit-min-local-y-span 15 \
  --hit-min-local-detections 4 \
  --hit-reject-same-directional-x-displacement 30
```

This keeps the candidate generator mostly recall-oriented, but removes pass-through candidates whose local x displacement does not look like a contact turn.
