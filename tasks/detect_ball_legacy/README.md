# detect_ball_legacy

Runs the imported legacy `UNet(27)` ball-tracking model on a video.

The model consumes a 9-frame RGB window resized to `640x360`, stacked into 27 input channels. It now defaults to a belief-heatmap temporal decoder (`--decoder belief`) that keeps a 2D location belief internally, predicts it forward with motion and gravity priors, and corrects it with each new UNet heatmap instead of collapsing immediately to a raw argmax.

```bash
docker compose run --rm app python tasks/detect_ball_legacy/run.py --max-frames 20
```

Use the older top-k + point-tracker decoder:

```bash
docker compose run --rm app python tasks/detect_ball_legacy/run.py --decoder track --max-frames 20
```

Revert to the previous single-frame argmax decoder:

```bash
docker compose run --rm app python tasks/detect_ball_legacy/run.py --decoder argmax --max-frames 20
```

Run a later segment:

```bash
docker compose run --rm app python tasks/detect_ball_legacy/run.py --start-frame 1000 --max-frames 20
```

Run a longer segment while saving debug images every 30 frames:

```bash
docker compose run --rm app python tasks/detect_ball_legacy/run.py --start-frame 900 --max-frames 1000 --debug-stride 30
```

GPU:

```bash
docker compose -f compose.yaml -f compose.gpu.yaml run --rm app python tasks/detect_ball_legacy/run.py --device cuda --max-frames 20
```

Useful decoder controls:

- `--candidate-threshold`: minimum heatmap score for top-k candidates before temporal decoding
- `--top-k`: number of heatmap peaks kept per frame
- `--peak-radius`: local suppression radius between selected peaks
- `--track-max-distance`: maximum allowed movement to associate a peak to the current track
- `--track-distance-weight`: motion penalty against candidate score during association
- `--belief-gravity`: per-frame vertical acceleration prior for the belief decoder
- `--belief-prior-blur`: diffusion strength applied after the belief state is moved forward
- `--belief-measurement-floor`: minimum likelihood mixed into each frame so the belief can survive short misses
