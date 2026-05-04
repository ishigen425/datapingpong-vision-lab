# annotate_bounce_video

Runs the lightweight softmax event classifier on ball coordinates and renders a video with `Bound!` text around predicted bounce frames.

```bash
docker compose run --rm app python tasks/annotate_bounce_video/run.py
```

Default input is the local DJI sample:

- video: `data/raw/DJI_0056_001-001.MP4`
- ball coordinates: `data/annotations/ball_tracking/DJI_0056_001_predictions.json`
- model: `models/lightweight_events/openttgames_softmax.json`
- output: `outputs/annotate_bounce_video/DJI_0056_001_bound.mp4`
