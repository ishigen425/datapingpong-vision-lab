# Event Detection Experiments

This document records the current coordinate-only event detection results so later changes can be compared against a stable baseline.

## Dataset And Split

- Dataset: OpenTTGames markup from `https://lab.osai.ai/`
- Imported files: markup zip files only, no videos
- Normalized annotations:
  - `data/annotations/openttgames/ball_positions.jsonl`
  - `data/annotations/openttgames/events.jsonl`
- OpenTTGames license: CC BY-NC-SA 4.0
- Lightweight ML split:
  - train: `game_1`, `game_2`, `game_3`, `game_4`, `game_5`
  - test: `test_1`, `test_2`, `test_3`, `test_4`, `test_5`, `test_6`, `test_7`
- Matching tolerance: `+-4` frames
- NMS window: `8` frames

## Results

## 2026-05-09 Bounce Detector Update

The OpenTTGames bounce detector was promoted from the coordinate-only rule baseline to a lightweight ML detector.

Rule baseline on OpenTTGames all items:

- precision: `0.606`
- recall: `0.878`
- F1: `0.717`

Current learned bounce detector on the OpenTTGames game/test split:

- method: coordinate-window features + numpy softmax regression
- training classes: `none`, `bounce`, `net_hit`
- emitted/evaluated event: `bounce`
- precision: `0.908`
- recall: `0.983`
- F1: `0.944`

Command:

```bash
docker compose run --rm app python tasks/train_event_classifier/run.py \
  --prediction-events bounce \
  --model-output models/lightweight_events/openttgames_bounce_detector_softmax.json \
  --predictions-output outputs/train_event_classifier/bounce_predictions.json \
  --summary-output outputs/train_event_classifier/bounce_summary.json
```

Interpretation:

- The rule baseline remains useful as a sanity check, but the learned detector is now the preferred OpenTTGames bounce baseline.
- Keeping `net_hit` as a separate training class helps distinguish table bounces from other sharp trajectory events, even when only `bounce` is emitted.
- This does not validate racket-hit detection. Local DJI inferred `hit` labels should still not be used as supervised train/test labels.

| Method | Data / Split | Classes | Precision | Recall | F1 | Notes |
| --- | --- | --- | ---: | ---: | ---: | --- |
| Rule baseline | local `DJI_0056_001` | bounce | 0.426 | 0.510 | 0.464 | `tasks/detect_events_from_ball`, threshold 0.35 |
| Rule baseline | local `DJI_0056_001` | hit | 0.369 | 0.369 | 0.369 | threshold 0.65 |
| X-flip rule + table gate | local `DJI_0056_001` | bounce | 0.819 | 0.671 | 0.738 | `smooth_window=3`, NMS 3, hit threshold 0.45 |
| X-flip rule + table gate | local `DJI_0056_001` | hit | 0.390 | 0.574 | 0.464 | Main cue is horizontal velocity reversal |
| X-flip + table-prior hit rule | local `DJI_0056_001` | bounce | 0.805 | 0.695 | 0.746 | Separate hit smoothing, table prior, arbitration window 4 |
| X-flip + table-prior hit rule | local `DJI_0056_001` | hit | 0.384 | 0.602 | 0.469 | Main cue is horizontal velocity reversal; recall improved |
| X-flip + table-prior + same-direction reject | local `DJI_0056_001` | bounce | 0.793 | 0.708 | 0.748 | `x35/y15`, reject strong same-direction hit candidates |
| X-flip + table-prior + same-direction reject | local `DJI_0056_001` | hit | 0.641 | 0.510 | 0.568 | Higher precision hit setting; predicted hits reduced to `198` |
| Softmax + table gate | local `DJI_0056_001` | bounce | 0.675 | 0.675 | 0.675 | OpenTTGames softmax, center prior, manual table polygon, threshold 0.5 |
| Rule baseline | OpenTTGames all items | bounce | 0.606 | 0.878 | 0.717 | Coordinate-only rule detector, no training |
| Softmax regression, bounce output only | OpenTTGames game/test | bounce | 0.908 | 0.983 | 0.944 | Trained with `none/bounce/net_hit`, emitted/evaluated only `bounce` |
| Softmax regression | OpenTTGames game/test | bounce | 0.915 | 0.983 | 0.948 | 39 scale-normalized coordinate-window features |
| Softmax regression | OpenTTGames game/test | net_hit | 0.388 | 0.967 | 0.553 | Recall is high, false positives are still high |
| Softmax regression | OpenTTGames game/test | micro avg for bounce/net_hit | 0.580 | 0.976 | 0.728 | Current best lightweight ML run |

## Commands

Rule baseline on local legacy annotations:

```bash
docker compose run --rm app python tasks/detect_events_from_ball/run.py \
  --input data/annotations/ball_tracking/DJI_0056_001_predictions.json \
  --output outputs/detect_events_from_ball/events.json

docker compose run --rm app python tasks/evaluate_event_detection/run.py \
  --reference data/annotations/events/bounce_and_hit_frames.json \
  --predictions outputs/detect_events_from_ball/events.json \
  --summary outputs/evaluate_event_detection/summary.json
```

Rule baseline on OpenTTGames bounce labels:

```bash
docker compose run --rm app python tasks/detect_events_from_ball/run.py \
  --input data/annotations/openttgames/ball_positions.jsonl \
  --output outputs/detect_events_from_ball/openttgames_events.json

docker compose run --rm app python tasks/evaluate_event_detection/run.py \
  --reference data/annotations/openttgames/events.jsonl \
  --predictions outputs/detect_events_from_ball/openttgames_events.json \
  --summary outputs/evaluate_event_detection/openttgames_summary.json \
  --events bounce
```

Lightweight ML current best run:

```bash
docker compose run --rm app python tasks/train_event_classifier/run.py
```

OpenTTGames bounce detector run that trains with `net_hit` as a separate class but emits/evaluates only `bounce`:

```bash
docker compose run --rm app python tasks/train_event_classifier/run.py \
  --prediction-events bounce \
  --model-output models/lightweight_events/openttgames_bounce_detector_softmax.json \
  --predictions-output outputs/train_event_classifier/bounce_predictions.json \
  --summary-output outputs/train_event_classifier/bounce_summary.json
```

Local DJI table-gated debug render with GPU video encode:

```bash
docker compose -f compose.yaml -f compose.gpu.yaml run --rm app \
  python tasks/annotate_bounce_video/run.py \
  --table-geometry data/annotations/table_geometry/DJI_0056_001.json \
  --threshold 0.5 \
  --video-codec h264_nvenc \
  --output outputs/annotate_bounce_video/DJI_0056_001_bound_table_nvenc.mp4 \
  --events-output outputs/annotate_bounce_video/bounce_events_table_nvenc.json \
  --subtitle-output outputs/annotate_bounce_video/bounce_events_table_nvenc.ass

docker compose run --rm app python tasks/evaluate_event_detection/run.py \
  --reference data/annotations/events/bounce_and_hit_frames.json \
  --predictions outputs/annotate_bounce_video/bounce_events_table_nvenc.json \
  --summary outputs/annotate_bounce_video/bounce_eval_table_nvenc_summary.json \
  --events bounce
```

Local DJI x-velocity-flip hit rule:

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

docker compose run --rm app python tasks/evaluate_event_detection/run.py \
  --reference data/annotations/events/bounce_and_hit_frames.json \
  --predictions outputs/detect_events_from_ball/local_table_xflip_hit_events.json \
  --summary outputs/evaluate_event_detection/local_table_xflip_hit_summary.json \
  --events bounce hit
```

Local DJI table-prior hit/bounce render:

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

docker compose run --rm app python tasks/evaluate_event_detection/run.py \
  --reference data/annotations/events/bounce_and_hit_frames.json \
  --predictions outputs/detect_events_from_ball/local_table_prior_motion_x30_y10_events.json \
  --summary outputs/evaluate_event_detection/local_table_prior_motion_x30_y10_summary.json \
  --events bounce hit

docker compose -f compose.yaml -f compose.gpu.yaml run --rm app \
  python tasks/annotate_events_video/run.py \
  --events outputs/detect_events_from_ball/local_table_prior_motion_x30_y10_events.json \
  --video-codec h264_nvenc \
  --output outputs/annotate_events_video/DJI_0056_001_events_table_motion_x30_y10_nvenc.mp4 \
  --subtitle-output outputs/annotate_events_video/events_table_motion_x30_y10_nvenc.ass
```

Compared with the unfiltered table-prior hit run, the balanced `x30/y10` local-motion gate reduces predicted hits from `391` to `249`. Hit precision improves from `0.384` to `0.550`; hit recall drops from `0.602` to `0.550`; hit F1 improves from `0.469` to `0.550`. The stricter `x40/y10` setting improves hit precision to `0.579` but makes hit labels too sparse by visual inspection. The optional `--hit-min-directional-x-displacement 8` check is currently too strict on this sample (`hit F1 0.341`).

Local DJI higher-precision hit run using same-direction rejection:

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

docker compose run --rm app python tasks/evaluate_event_detection/run.py \
  --reference data/annotations/events/bounce_and_hit_frames.json \
  --predictions outputs/detect_events_from_ball/local_table_prior_motion_x35_y15_same30_events.json \
  --summary outputs/evaluate_event_detection/local_table_prior_motion_x35_y15_same30_summary.json \
  --events bounce hit
```

This run reduces predicted hits from `249` to `198`. Hit precision improves from `0.550` to `0.641`; hit recall drops from `0.550` to `0.510`; hit F1 improves from `0.550` to `0.568`. Bounce is nearly unchanged (`F1 0.748`).

Default ML settings:

- model: numpy softmax regression with standardization
- labels: `none`, `bounce`, `net_hit`
- features: 39 coordinate-window and kinematic features normalized by frame size
- frame size: `1280x720`
- optional table geometry: hand-annotated corner JSON adds 7 table-relative features and can gate bounce candidates to the table region
- threshold: `0.35`
- positive radius: `+-2` frames
- negative margin: `12` frames
- negative ratio: `1.5`
- epochs: `700`
- learning rate: `0.08`
- L2: `0.001`

Bounce-only binary classifier check:

```bash
docker compose run --rm app python tasks/train_bounce_classifier/run.py
```

This task is useful as an ablation, but it is not the current best OpenTTGames bounce detector. The best binary run so far reached bounce F1 around `0.915`, while the multiclass-trained bounce-only output reaches F1 `0.944`. The likely reason is that keeping `net_hit` as its own class provides a useful hard distinction from table bounces.

## Failed Or Weaker Runs

Including `empty` as a fourth class destabilized the first multiclass check:

| Method | Classes | Precision | Recall | F1 | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| Softmax regression | bounce | null | 0.000 | null | Predicted no bounce events at threshold 0.35 |
| Softmax regression | net_hit | 0.250 | 0.006 | 0.011 | Almost no net_hit detections |
| Softmax regression | empty | 0.091 | 0.711 | 0.162 | Many false positives |
| Softmax regression | micro avg | 0.093 | 0.071 | 0.080 | Not usable |

For now, `empty` should be treated as a separate non-contact/segment label rather than trained in the same contact-event classifier.

## Interpretation

- The rule baseline is useful for sanity checks, but not practical as a final detector.
- The lightweight ML bounce detector is already materially better than rules on the OpenTTGames game/test split.
- For the current OpenTTGames bounce detector, train with `none`, `bounce`, and `net_hit`, then emit only `bounce`. A purely binary `none/bounce` setup performed worse in the first ablation.
- Adding a hand-measured table polygon and applying it after scaling local ball coordinates from `640x360` to the `1920x1080` video frame improves local DJI bounce F1 from `0.464` to `0.675`.
- Explicit horizontal velocity reversal improves local DJI hit F1 from `0.369` to `0.464`, but precision is still weak because several non-racket trajectory changes produce similar x-flips.
- The table-prior hit rule uses separate smoothing for bounce (`3`) and hit (`7`), boosts hit candidates near/outside table edges, and arbitrates close bounce/hit conflicts by table zone. It improves local hit recall to `0.602` and hit F1 to `0.469`.
- Adding a local-motion quality gate around hit candidates is useful for poor ball visibility and non-rally sections. Requiring a `+-6` frame local span of at least `30px` in x, `10px` in y, and `4` detected points balances the local DJI hit count at `249` predictions and improves hit F1 to `0.550`.
- Rejecting hit candidates whose pre/post x displacement continues strongly in the same direction is a useful precision-oriented second-stage filter. The current `x35/y15/same30` run improves local hit precision to `0.641` and F1 to `0.568`, at the cost of recall.
- Visual inspection of the table-gated render looked close to 80% usable, but remaining false positives/negatives still need threshold and table-relative model work.
- The net_hit detector needs stricter precision work before practical use. The next candidates are class-specific thresholds, richer trajectory features around net-hit candidates, and a held-out threshold sweep.
- The reported ML numbers are OpenTTGames-only and should not be interpreted as racket-hit performance on the local DJI annotation, because OpenTTGames labels do not include the same `hit` class.
