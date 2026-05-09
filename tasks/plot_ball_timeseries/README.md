# plot_ball_timeseries

Writes SVG plots of ball `x` and `y` coordinates over time, with optional reference and predicted event markers.

Full-video plot:

```bash
docker compose run --rm app python tasks/plot_ball_timeseries/run.py \
  --ball data/annotations/ball_tracking/DJI_0056_001_predictions.json \
  --events outputs/detect_events_from_ball/local_table_prior_motion_x30_y10_events.json \
  --reference data/annotations/events/bounce_and_hit_frames.json \
  --table-geometry data/annotations/table_geometry/DJI_0056_001.json
```

Default output:

```text
outputs/plot_ball_timeseries/DJI_0056_001_timeseries.svg
```

When `--table-geometry` is provided, the SVG includes a second panel with table-relative `table_x` and `table_y` coordinates. The dashed horizontal lines mark the `0..1` table bounds, and green dots mark frames where the ball is inside the table polygon.

Plot only false-positive and false-negative windows from the event diagnosis CSV:

```bash
docker compose run --rm app python tasks/plot_ball_timeseries/run.py \
  --events outputs/detect_events_from_ball/local_table_prior_motion_x30_y10_events.json \
  --reference data/annotations/events/bounce_and_hit_frames.json \
  --table-geometry data/annotations/table_geometry/DJI_0056_001.json \
  --diagnostics outputs/diagnose_event_detection/details.csv \
  --diagnostic-status false_positive false_negative \
  --window-radius 45 \
  --max-windows 40
```

Window SVGs are written under:

```text
outputs/plot_ball_timeseries/windows/
```

Use these plots to decide whether an error is caused by ball tracking, event scoring, event arbitration, or annotation tolerance before changing the model.
