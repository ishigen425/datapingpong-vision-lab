# estimate_pose

Runs MediaPipe Pose on a video and writes per-frame pose landmarks as JSON.

```bash
docker compose run --rm app python tasks/estimate_pose/run.py
```

Default input is the local DJI sample:

- video: `data/raw/DJI_0056_001-001.MP4`
- model: `models/checkpoints/mediapipe/pose_landmarker_full.task`
- output: `outputs/estimate_pose/DJI_0056_001_pose.json`

On the first run, the task downloads the default MediaPipe pose landmarker bundle into the ignored `models/checkpoints/mediapipe/` directory if it is missing.

The output stores up to two detected players per frame, labeled `left` and `right` from image position, plus normalized image landmarks, optional world landmarks, and task metadata. Use it as the input to `tasks/extract_pose_features/run.py` when you want player-wise body-relative features for later shot-event or stroke-style experiments.
