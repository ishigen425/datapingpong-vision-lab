# evaluate_ball_predictions

Compares `tasks/detect_ball_legacy` output against the imported legacy coordinate JSON.

```bash
docker compose run --rm app python tasks/evaluate_ball_predictions/run.py
```

By default this compares `reference_frame = prediction_frame - 8`, matching the imported legacy 9-frame-window output.

Outputs:

- `outputs/evaluate_ball_predictions/summary.json`
- `outputs/evaluate_ball_predictions/frame_diffs.csv`
