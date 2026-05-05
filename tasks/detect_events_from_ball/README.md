# detect_events_from_ball

Scores bounce and hit probabilities from ball coordinates using trajectory features only.

```bash
docker compose run --rm app python tasks/detect_events_from_ball/run.py
```

Inputs can be the legacy ball JSON, detector prediction JSON, OpenTTGames frame-to-coordinate JSON, or normalized JSONL from `tasks/import_openttgames`.
Pass `--table-geometry data/annotations/table_geometry/<video>.json` to keep bounce candidates on the annotated table region.

Default thresholds are tuned for the imported local `DJI_0056_001` annotation:

- bounce: `0.35`
- hit: `0.65`
- NMS window: `8` frames
