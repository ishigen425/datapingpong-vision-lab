# evaluate_rally_proposals

Evaluates the current rule-based rally proposal decision against manually labeled proposal rows.

```bash
docker compose run --rm app python tasks/evaluate_rally_proposals/run.py \
  --proposals outputs/detect_rallies/DJI_0056_001_rally_proposals.json
```

Expected workflow:

1. Fill `label` in the proposals JSON.
2. Run this task.
3. Inspect:
   - overall precision / recall / F1 of `rule_keep`
   - `by_label` counts
   - false positives / false negatives in `details.json`

This is the main immediate use of the labeled proposal file before there is enough data to justify a real ML model.
