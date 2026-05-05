# extract_stroke_details

Extracts per-hit stroke details from hit events, ball tracks, and pose features.

```bash
docker compose run --rm app python tasks/extract_stroke_details/run.py
```

Current output fields per hit:

- `player_role`
- `striking_arm`
- `contact_side`
- `stroke_side`
- `confidence`
- `ball_wrist_distance`
- `wrist_speed`

This task is intentionally independent from rally segmentation so the stroke logic can be rebuilt later without disturbing rally detection.
