# datapingpong-vision-lab

Table tennis vision workspace for separating ball tracking from event detection.

The current repository contains:

- legacy PyTorch ball-tracking code imported from older notebook work
- coordinate-only rule-based event detection
- lightweight softmax-regression event classification from ball trajectories
- OpenTTGames markup import/normalization utilities
- evaluation tasks for ball coordinates and event frames
- video rendering utilities for visual inspection

Large videos, extracted datasets, checkpoints, model outputs, and rendered videos are intentionally kept out of Git.

## Current Status

The project is useful for experimentation and evaluation, not yet for production event detection.

Current coordinate-only event detection records are in [docs/event_detection_experiments.md](docs/event_detection_experiments.md).

Key results so far:

| Method | Dataset / Split | Event | Precision | Recall | F1 |
| --- | --- | --- | ---: | ---: | ---: |
| Rule baseline | local DJI annotation | bounce | 0.426 | 0.510 | 0.464 |
| Rule baseline | local DJI annotation | hit | 0.369 | 0.369 | 0.369 |
| Softmax + table gate | local DJI annotation | bounce | 0.675 | 0.675 | 0.675 |
| Rule baseline | OpenTTGames all items | bounce | 0.606 | 0.878 | 0.717 |
| Softmax regression | OpenTTGames game/test | bounce | 0.915 | 0.983 | 0.948 |
| Softmax regression | OpenTTGames game/test | net_hit | 0.388 | 0.967 | 0.553 |
| Softmax regression | OpenTTGames game/test | bounce/net_hit micro avg | 0.580 | 0.976 | 0.728 |

Important caveat: OpenTTGames labels `bounce`, `net_hit`, and `empty`; it does not directly match the local DJI `hit` label. The OpenTTGames softmax model performs poorly when applied directly to the local DJI video, so that path should be treated as visualization/debugging rather than a reliable detector.

## Repository Layout

```text
archive/              # Downloaded/imported zip files, ignored by Git
data/
  raw/                # Source videos, ignored by Git
  annotations/        # Small tracked annotations plus ignored normalized OpenTTGames data
docs/                 # Notes, experiment records, and TODOs
models/
  checkpoints/        # Legacy model checkpoints, ignored by Git
  lightweight_events/ # Trained lightweight event models, ignored by Git
notebooks/
  legacy/             # Reference notebooks from previous work
outputs/              # Generated predictions, summaries, frames, and videos, ignored by Git
src/datapingpong/     # Shared Python package code
tasks/                # Independently runnable task entrypoints
tests/                # Focused tests for event logic and lightweight ML
```

## Requirements

- Docker
- Docker Compose
- NVIDIA Container Toolkit, only when using GPU

The Docker image installs PyTorch, OpenCV, pandas, pytest, and ffmpeg.

## Build

```bash
docker compose build
```

For GPU-capable hosts:

```bash
docker compose -f compose.yaml -f compose.gpu.yaml build
```

## Smoke Test

```bash
docker compose run --rm app
```

This runs `tasks/verify_torch/run.py` and prints PyTorch/CUDA status.

Run the focused test suite:

```bash
docker compose run --rm app pytest -q tests/test_events.py tests/test_event_ml.py
```

## Data

### Local DJI Sample

The local sample video is expected at:

```text
data/raw/DJI_0056_001-001.MP4
```

Existing local coordinate and event annotations:

```text
data/annotations/ball_tracking/DJI_0056_001_predictions.json
data/annotations/events/bounce_and_hit_frames.json
```

### OpenTTGames

Download and normalize OpenTTGames markup zip files only:

```bash
python3 tasks/import_openttgames/run.py
```

Useful dry run:

```bash
python3 tasks/import_openttgames/run.py --dry-run
```

Outputs:

```text
archive/openttgames/markup/*.zip
data/openttgames/markup/
data/annotations/openttgames/ball_positions.jsonl
data/annotations/openttgames/events.jsonl
```

OpenTTGames is licensed under CC BY-NC-SA 4.0. Keep downstream use non-commercial unless separate permission exists.

## Main Workflows

### Legacy Ball Detector

Run the imported DL ball detector on the sample video:

```bash
docker compose run --rm app python tasks/detect_ball_legacy/run.py --max-frames 20
```

Compare detector output against the imported coordinate JSON:

```bash
docker compose run --rm app python tasks/evaluate_ball_predictions/run.py
```

### Rule-Based Event Detection

Detect bounce/hit-like events from coordinates:

```bash
docker compose run --rm app python tasks/detect_events_from_ball/run.py
```

Optionally gate bounce candidates to a hand-annotated table polygon:

```bash
docker compose run --rm app python tasks/detect_events_from_ball/run.py \
  --table-geometry data/annotations/table_geometry/<video>.json
```

Evaluate predicted event frames:

```bash
docker compose run --rm app python tasks/evaluate_event_detection/run.py
```

### Lightweight Event Classifier

Train and evaluate the coordinate-window softmax regression classifier:

```bash
docker compose run --rm app python tasks/train_event_classifier/run.py
```

Pass `--table-geometry` to add table-relative features such as normalized table position, inside/outside state, edge margin, and net distance.

Default training split:

- train: `game_1` to `game_5`
- test: `test_1` to `test_7`
- labels: `none`, `bounce`, `net_hit`

The default excludes `empty` because the first multiclass check was unstable. See [docs/event_detection_experiments.md](docs/event_detection_experiments.md).

### Render Bounce Debug Video

Render `Bound!` around softmax-predicted bounce frames in the local DJI sample:

```bash
docker compose run --rm app python tasks/annotate_bounce_video/run.py
```

When `--table-geometry` is provided, rendering filters predicted bounces to the table region. Models trained with table-relative features also consume those features automatically.

Default outputs:

```text
outputs/annotate_bounce_video/DJI_0056_001_bound.mp4
outputs/annotate_bounce_video/bounce_events.json
outputs/annotate_bounce_video/bounce_events.ass
outputs/annotate_bounce_video/bounce_eval_summary.json
```

This is a debug visualization. The OpenTTGames-trained softmax bounce model currently transfers poorly to the local DJI sample.

## Useful Commands

Open a shell:

```bash
docker compose run --rm app bash
```

Run with GPU override:

```bash
docker compose -f compose.yaml -f compose.gpu.yaml run --rm app
```

Inspect Git-tracked changes:

```bash
git status --short
git diff --stat
```

## Git And Artifacts

Do not commit:

- videos under `data/raw/`
- downloaded/extracted OpenTTGames data
- checkpoints under `models/checkpoints/`
- trained lightweight model JSON under `models/lightweight_events/`
- generated outputs under `outputs/`

These paths are ignored in `.gitignore`. Keep code, docs, tests, and small hand-written annotations in Git.
