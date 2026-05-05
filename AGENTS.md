# AGENTS.md

Guidance for coding agents working in this repository.

## Project Goal

This repository explores table-tennis vision pipelines where ball tracking and event detection are separated. Prefer modular coordinate-first workflows over a single end-to-end model unless the user explicitly asks for an end-to-end approach.

## Environment

- Work from the repository root.
- Use Docker for normal execution:

```bash
docker compose run --rm app <command>
```

- Build after Dockerfile or dependency changes:

```bash
docker compose build
```

- Use the GPU override only when needed:

```bash
docker compose -f compose.yaml -f compose.gpu.yaml run --rm app <command>
```

### GPU Video Encoding With ffmpeg

Use the GPU compose override and pass `h264_nvenc` to task scripts that expose `--video-codec`:

```bash
docker compose -f compose.yaml -f compose.gpu.yaml run --rm app \
  python tasks/annotate_events_video/run.py \
  --events outputs/detect_events_from_ball/local_table_prior_motion_x30_y10_events.json \
  --video-codec h264_nvenc
```

The task writes large ignored artifacts under `outputs/`. If NVENC is unavailable on the host, rerun the same task without the GPU override and use `--video-codec libx264`.

## Data And Artifact Rules

Do not commit large or generated files.

Ignored artifact locations include:

- `archive/openttgames/`
- `data/raw/`
- `data/openttgames/`
- `data/annotations/openttgames/`
- `models/checkpoints/`
- `models/lightweight_events/`
- `outputs/`

The local DJI video and OpenTTGames downloads may exist in the workspace, but they are working data, not source files. Keep experiment summaries in `docs/` when results need to be preserved.

OpenTTGames is CC BY-NC-SA 4.0. Do not describe outputs from that dataset as commercially usable.

## Code Organization

- Shared reusable code lives under `src/datapingpong/`.
- Runnable experiments and one-off workflows live under `tasks/<task_name>/run.py`.
- Each task should have a short `README.md` explaining inputs, command, and outputs.
- Tests live under `tests/`.
- Experiment records live under `docs/`.

Current important modules:

- `src/datapingpong/events/trajectory.py`: rule-based trajectory event scores
- `src/datapingpong/events/features.py`: coordinate-window features
- `src/datapingpong/events/ml.py`: numpy softmax regression
- `src/datapingpong/events/io.py`: annotation and prediction IO helpers

## Validation

Before finishing code changes, run focused tests:

```bash
docker compose run --rm app pytest -q tests/test_events.py tests/test_event_ml.py
```

For Dockerfile changes, also run:

```bash
docker compose build
```

For video-rendering changes, verify ffmpeg path through the task:

```bash
docker compose run --rm app python tasks/annotate_bounce_video/run.py
```

That command writes a large ignored MP4 under `outputs/`.

## Experiment Recording

When comparing methods, record:

- method name
- dataset and split
- labels/classes
- threshold and NMS settings
- precision, recall, F1
- command used to reproduce the run
- important caveats

Use `docs/event_detection_experiments.md` for event-detection results.

## Current Caveats

- Rule-based detection is a baseline, not production quality.
- OpenTTGames-trained softmax regression works well for OpenTTGames bounce on the game/test split, but transfers poorly to the local DJI sample.
- `empty` is excluded from the default lightweight classifier because the first multiclass run was unstable.
- Local DJI `hit` and OpenTTGames `net_hit` are not equivalent labels.

## Style

- Keep task scripts explicit and runnable from repo root.
- Prefer standard library plus existing dependencies unless a new dependency is clearly justified.
- Keep generated model files and outputs out of Git.
- Update README or task docs when adding a new workflow.
