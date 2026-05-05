# extract_pose_features

Builds body-relative per-frame features from `tasks/estimate_pose` JSON output.

```bash
docker compose run --rm app python tasks/extract_pose_features/run.py
```

Default inputs and outputs:

- pose: `outputs/estimate_pose/DJI_0056_001_pose.json`
- output: `outputs/extract_pose_features/DJI_0056_001_pose_features.json`

Current features are exported separately for `left` and `right` players and focus on upper-body motion that should be reusable for later hit timing and stroke-type experiments: shoulder/hip centers, body scale, arm-relative wrist and elbow positions, wrist velocity, arm angles, and simple body tilt signals.
