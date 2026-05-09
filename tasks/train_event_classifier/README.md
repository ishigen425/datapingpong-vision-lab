# train_event_classifier

Trains a lightweight softmax-regression event classifier from ball coordinates.

The default split trains on `game_1` through `game_5` and evaluates on `test_1` through `test_7`.
The default labels are `none`, `bounce`, and `net_hit`; `empty` is excluded because it is not a physical contact event and destabilized the first multiclass check.
Coordinate-window offsets, velocities, rule speeds, and absolute positions are normalized by the configured frame size. The OpenTTGames default is `1280x720`.
Pass `--table-geometry data/annotations/table_geometry/<video>.json` to add table-relative features for a single-camera item.

```bash
docker compose run --rm app python tasks/train_event_classifier/run.py
```

Outputs:

- `models/lightweight_events/openttgames_softmax.json`
- `outputs/train_event_classifier/predictions.json`
- `outputs/train_event_classifier/summary.json`

To train with `net_hit` as a separate class but emit/evaluate only the stronger `bounce` detector:

```bash
docker compose run --rm app python tasks/train_event_classifier/run.py \
  --prediction-events bounce \
  --model-output models/lightweight_events/openttgames_bounce_detector_softmax.json \
  --predictions-output outputs/train_event_classifier/bounce_predictions.json \
  --summary-output outputs/train_event_classifier/bounce_summary.json
```
