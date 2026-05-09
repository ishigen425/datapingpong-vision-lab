# train_event_candidate_classifier

Trains a lightweight classifier that accepts or rejects rule-generated event candidates.

This is the first local ML pass for the DJI sample. The rule detector remains the candidate generator; the classifier is a second-stage filter over candidate rows using:

- candidate event type and probability
- local trajectory diagnostics
- frame-level rule scores
- image position
- optional table-relative features

Default command:

```bash
docker compose run --rm app python tasks/train_event_candidate_classifier/run.py \
  --candidate-events outputs/detect_events_from_ball/local_table_prior_motion_x30_y10_events.json \
  --table-geometry data/annotations/table_geometry/DJI_0056_001.json
```

Outputs:

- `models/lightweight_events/local_candidate_softmax.json`
- `outputs/train_event_candidate_classifier/predictions.json`
- `outputs/train_event_candidate_classifier/summary.json`
- `outputs/train_event_candidate_classifier/training_rows.csv`

The default split is temporal block based. Blocks alternate between train and test so adjacent frames are not randomly mixed. Treat the metrics as an early local experiment, not production validation.

The classifier is used as an event-specific acceptance score, not as a hard multiclass argmax by default. Current defaults use:

- bounce accept threshold: `0.30`
- hit accept threshold: `0.15`

Pass `--require-argmax` only when you want stricter, lower-recall filtering.
