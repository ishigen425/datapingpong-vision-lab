# detect_ball_legacy

Runs the imported legacy `UNet(27)` ball-tracking model on a video.

The model consumes a 9-frame RGB window resized to `640x360`, stacked into 27 input channels. It writes JSON predictions and debug frames with the predicted ball location.

```bash
docker compose run --rm app python tasks/detect_ball_legacy/run.py --max-frames 20
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
