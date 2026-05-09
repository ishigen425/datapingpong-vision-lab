# train_torch_event_classifier

Trains a heavier PyTorch MLP event classifier from OpenTTGames ball-coordinate features, then exports JSON weights that existing event annotation tasks can load.

CPU smoke run:

```bash
docker compose run --rm app python tasks/train_torch_event_classifier/run.py --epochs 5
```

GPU run:

```bash
docker compose -f compose.yaml -f compose.gpu.yaml run --rm app \
  python tasks/train_torch_event_classifier/run.py \
  --prediction-events bounce \
  --architecture mlp \
  --hidden-units 128 64 \
  --epochs 250
```

Transformer run:

```bash
docker compose -f compose.yaml -f compose.gpu.yaml run --rm app \
  python tasks/train_torch_event_classifier/run.py \
  --prediction-events bounce \
  --architecture transformer \
  --sequence-radius 8 \
  --transformer-d-model 96 \
  --transformer-heads 4 \
  --transformer-layers 3 \
  --transformer-feedforward 192 \
  --epochs 80
```

Validation uses the OpenTTGames test split only. Local DJI output from this model should be treated as visual review material, not validation data.
