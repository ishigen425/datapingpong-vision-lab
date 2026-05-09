# train_bounce_classifier

Trains a lightweight coordinate-window classifier for `bounce` only.

This task uses OpenTTGames `bounce` labels and avoids `net_hit`/local `hit` labels. It is the preferred ML baseline when improving bounce detection.

```bash
docker compose run --rm app python tasks/train_bounce_classifier/run.py
```

Default split:

- train: `game_1` through `game_5`
- test: `test_1` through `test_7`

Outputs:

- `models/lightweight_events/openttgames_bounce_softmax.json`
- `outputs/train_bounce_classifier/predictions.json`
- `outputs/train_bounce_classifier/summary.json`

The task sweeps thresholds on the test split and records the best F1 threshold in the summary. The model is intentionally lightweight so it can act as a reliable coordinate-first bounce detector before rally segmentation work.

By default, `net_hit` frames are used as hard negatives. They are not predicted by this task, but they help the binary model separate table bounces from other sharp trajectory events.
