# Table Geometry Annotations

Small hand-written table corner annotations can live here. Use one JSON file per video or dataset item.

Corner order is image-space table corners:

```json
{
  "image_width": 1920,
  "image_height": 1080,
  "corners": {
    "far_left": [520, 310],
    "far_right": [1390, 305],
    "near_right": [1710, 780],
    "near_left": [260, 790]
  }
}
```

The loader maps those corners to normalized table coordinates:

- `far_left`: `(0, 0)`
- `far_right`: `(1, 0)`
- `near_right`: `(1, 1)`
- `near_left`: `(0, 1)`

Do not add generated detection outputs here; keep those under `outputs/`.
