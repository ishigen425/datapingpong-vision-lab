# train_rally_start_classifier

Trains a lightweight softmax classifier from labeled rally-start proposal rows.

```bash
docker compose run --rm app python tasks/train_rally_start_classifier/run.py \
  --proposals outputs/detect_rallies/DJI_0056_001_rally_proposals.json
```

The proposals file must contain labeled rows in the `label` field. Expected labels are project-defined, for example:

- `true_rally`
- `handoff`
- `serve_miss_or_ace`

This task is intentionally lightweight and reuses the repository's existing numpy softmax regression implementation.
