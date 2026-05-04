# evaluate_event_detection

Evaluates predicted event frames against bounce/hit annotations with a frame tolerance.

```bash
docker compose run --rm app python tasks/evaluate_event_detection/run.py
```

By default it compares:

- reference: `data/annotations/events/bounce_and_hit_frames.json`
- predictions: `outputs/detect_events_from_ball/events.json`
- output: `outputs/evaluate_event_detection/summary.json`
