# datapingpong-vision-lab

Docker-based PyTorch workspace for table tennis vision experiments.

## Directory Layout

```text
archive/              # Original imported zip files, not tracked
data/
  raw/                # Source videos, not tracked
  annotations/        # Imported JSON annotations and coordinate/event data
models/
  checkpoints/        # Imported model weights, not tracked
notebooks/
  legacy/             # Reference notebooks from the previous workspace
outputs/
  legacy/             # Imported generated frames, videos, and jsonl outputs
src/datapingpong/     # Shared Python code
tasks/                # Independently runnable task entrypoints
```

The current useful imported code is under:

```text
src/datapingpong/models/legacy_ball_tracking/
src/datapingpong/tracking/legacy_detector.py
```

Try the imported DL ball detector on the sample video:

```bash
docker compose run --rm app python tasks/detect_ball_legacy/run.py --max-frames 20
```

Compare detector output against the imported legacy coordinate JSON:

```bash
docker compose run --rm app python tasks/evaluate_ball_predictions/run.py
```

Download and normalize OpenTTGames markup files only:

```bash
python3 tasks/import_openttgames/run.py
```

Detect bounce/hit-like events from ball coordinates without an end-to-end event model:

```bash
docker compose run --rm app python tasks/detect_events_from_ball/run.py
docker compose run --rm app python tasks/evaluate_event_detection/run.py
```

Train and evaluate the lightweight coordinate-based event classifier:

```bash
docker compose run --rm app python tasks/train_event_classifier/run.py
```

Current event-detection experiment results are tracked in `docs/event_detection_experiments.md`.

The OpenTTGames dataset is licensed as CC BY-NC-SA 4.0, so keep downstream use non-commercial unless you have separate permission.

The original large files are kept out of Git by `.gitignore`.

## Requirements

- Docker
- Docker Compose
- NVIDIA Container Toolkit, only when using GPU

## Build

```bash
docker compose build
```

## Run

```bash
docker compose run --rm app
```

This runs `tasks/verify_torch/run.py` and prints the PyTorch version, CUDA availability, selected device, and a small tensor calculation.

## Shell

```bash
docker compose run --rm app bash
```

## GPU Run

If the machine has an NVIDIA GPU and NVIDIA Container Toolkit:

```bash
docker compose -f compose.yaml -f compose.gpu.yaml run --rm app
```

Without the GPU override, the same image still runs on CPU.
