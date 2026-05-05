# TODO

## Docker Python Package Management

Resolved in the Dockerfile by copying the `uv` binary from `ghcr.io/astral-sh/uv`, creating `/opt/venv` with `--system-site-packages` so the official PyTorch image packages remain visible, and syncing runtime dependencies from `pyproject.toml` with `uv.lock`:

```bash
uv sync --locked --active --inexact
```

## Event Detection Handoff

Current local DJI coordinate-rule baseline:

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

- bounce: precision `0.796`, recall `0.708`, F1 `0.749`, predicted `216`
- hit: precision `0.550`, recall `0.550`, F1 `0.550`, predicted `249`

Rendered review video:

```text
outputs/annotate_events_video/DJI_0056_001_events_table_motion_x30_y10_nvenc.mp4
```

Notes for the next pass:

- `x40/y10` was too sparse visually even though hit precision improved.
- `--hit-min-directional-x-displacement 8` was too strict on this sample: hit dropped to `79` predictions and F1 `0.341`.
- Predicted hit events now include diagnostic fields: `local_x_span`, `local_y_span`, `local_detections`, `local_before_x_displacement`, and `local_after_x_displacement`.
- Next useful step is likely reviewing false positives/false negatives using those diagnostics, then deciding whether to add PoseEstimation/racket context or train a local lightweight classifier.
