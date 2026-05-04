# train_event_classifier

Trains a lightweight softmax-regression event classifier from ball coordinates.

The default split trains on `game_1` through `game_5` and evaluates on `test_1` through `test_7`.
The default labels are `none`, `bounce`, and `net_hit`; `empty` is excluded because it is not a physical contact event and destabilized the first multiclass check.

```bash
docker compose run --rm app python tasks/train_event_classifier/run.py
```

Outputs:

- `models/lightweight_events/openttgames_softmax.json`
- `outputs/train_event_classifier/predictions.json`
- `outputs/train_event_classifier/summary.json`
