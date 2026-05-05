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

| Method | Data / Split | Classes | Precision | Recall | F1 | Notes |
| --- | --- | --- | ---: | ---: | ---: | --- |
| Rule baseline | local `DJI_0056_001` | bounce | 0.426 | 0.510 | 0.464 | `tasks/detect_events_from_ball`, threshold 0.35 |
| Rule baseline | local `DJI_0056_001` | hit | 0.369 | 0.369 | 0.369 | threshold 0.65 |
| Softmax + table gate | local `DJI_0056_001` | bounce | 0.675 | 0.675 | 0.675 | OpenTTGames softmax, center prior, manual table polygon, threshold 0.5 |
| Rule baseline | OpenTTGames all items | bounce | 0.606 | 0.878 | 0.717 | Coordinate-only rule detector, no training |
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
- Adding a hand-measured table polygon and applying it after scaling local ball coordinates from `640x360` to the `1920x1080` video frame improves local DJI bounce F1 from `0.464` to `0.675`.
- Visual inspection of the table-gated render looked close to 80% usable, but remaining false positives/negatives still need threshold and table-relative model work.
- The net_hit detector needs stricter precision work before practical use. The next candidates are class-specific thresholds, richer trajectory features around net-hit candidates, and a held-out threshold sweep.
- The reported ML numbers are OpenTTGames-only and should not be interpreted as racket-hit performance on the local DJI annotation, because OpenTTGames labels do not include the same `hit` class.
