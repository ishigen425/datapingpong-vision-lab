# annotate_bounce_video

Runs the lightweight softmax event classifier on ball coordinates and renders a video with `BOUND!` text around predicted bounce frames.

```bash
docker compose run --rm app python tasks/annotate_bounce_video/run.py
```

Default input is the local DJI sample:

- video: `data/raw/DJI_0056_001-001.MP4`
- ball coordinates: `data/annotations/ball_tracking/DJI_0056_001_predictions.json`
- model: `models/lightweight_events/openttgames_softmax.json`
- output: `outputs/annotate_bounce_video/DJI_0056_001_bound.mp4`

Default scoring applies a broad center-of-frame spatial prior after the softmax model:

- threshold: `0.025`
- weak threshold: disabled by default
- NMS window: `8`
- spatial prior weight: `2.25`
- spatial prior sigma: `x=0.32`, `y=0.30`

Pass `--table-geometry data/annotations/table_geometry/<video>.json` to filter rendered bounces to the table region. If the model was trained with table-relative features, the task also supplies those features at prediction time.

Pass `--weak-threshold` below `--threshold` to render lower-confidence candidates as `weak bounce?` instead of dropping them. This is useful when reviewing whether bounce recall is high enough for rally proposal generation.

Pass `--serve-threshold` below `--threshold` to render lower-confidence candidates as `serve bounce?` only when the immediately preceding ball track looks like a serve toss: small x-span, upward y motion, then downward motion into the candidate. This raises recall only in serve-like context instead of lowering the threshold globally.
