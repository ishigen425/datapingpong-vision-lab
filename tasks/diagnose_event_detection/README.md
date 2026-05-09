# diagnose_event_detection

Builds a review table for event-detection false positives, false negatives, and matches.

```bash
docker compose run --rm app python tasks/diagnose_event_detection/run.py \
  --predictions outputs/detect_events_from_ball/local_table_prior_motion_x30_y10_events.json
```

Default inputs:

- reference: `data/annotations/events/bounce_and_hit_frames.json`
- predictions: `outputs/detect_events_from_ball/local_table_prior_motion_x30_y10_events.json`
- output directory: `outputs/diagnose_event_detection/`

Outputs:

- `summary.json`: per-event counts, precision, recall, F1, and diagnostic aggregates
- `details.json`: one row per true positive, false positive, and false negative
- `details.csv`: flat review table for sorting/filtering

Useful fields in `details.csv`:

- `status`: `true_positive`, `false_positive`, or `false_negative`
- `event`
- `frame`
- `matched_frame`
- `frame_error`
- `nearest_reference_frame`
- `nearest_prediction_frame`
- `probability`
- `bounce_probability`
- `hit_probability`
- `detected`
- `confidence`
- `speed`
- `acceleration`
- `turn_angle_degrees`
- hit diagnostics such as `local_x_span`, `local_y_span`, and `local_detections`

Use this before changing thresholds or models. If most false negatives have poor ball context, improve tracking first. If ball context is clean but the class is wrong, improve event scoring or add pose/racket features.
