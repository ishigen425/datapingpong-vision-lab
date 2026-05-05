# annotate_multimodal_video

Renders the local sample video with ball trajectory, bounce/hit events, pose landmarks, and pose feature summaries in one MP4.

```bash
docker compose run --rm app python tasks/annotate_multimodal_video/run.py
```

Default inputs:

- video: `data/raw/DJI_0056_001-001.MP4`
- ball: `data/annotations/ball_tracking/DJI_0056_001_predictions.json`
- events: `outputs/detect_events_from_ball/local_table_prior_motion_x30_y10_events.json`
- pose: `outputs/estimate_pose/DJI_0056_001_pose.json`
- pose features: `outputs/extract_pose_features/DJI_0056_001_pose_features.json`
- rallies: optional `outputs/detect_rallies/*.json`

For the imported local DJI legacy ball JSON, pass `--ball-frame-offset 4` so the ball and derived events line up with the center frame of the original 9-frame detector window.

Default output:

- `outputs/annotate_multimodal_video/DJI_0056_001_multimodal.mp4`

The overlay shows:

- recent ball trajectory trail and current ball location
- `BOUNCE` / `HIT` labels near the corresponding frames
- optional current rally panel with rally number, frame range, and event counts
- `left` / `right` upper-body pose skeletons from MediaPipe Pose
- a small pose-feature panel with current extracted values for both players

To inspect rally segmentation together with belief-tracked ball and pose overlays:

```bash
docker compose -f compose.yaml -f compose.gpu.yaml run --rm app \
  python tasks/annotate_multimodal_video/run.py \
  --ball outputs/detect_ball_legacy/DJI_0056_001_belief_predictions.json \
  --events outputs/detect_events_from_ball/DJI_0056_001_belief_events.json \
  --rallies outputs/detect_rallies/DJI_0056_001_belief_rallies.json \
  --output outputs/annotate_multimodal_video/DJI_0056_001_multimodal_belief_rallies.mp4 \
  --summary-output outputs/annotate_multimodal_video/DJI_0056_001_multimodal_belief_rallies_summary.json \
  --video-codec h264_nvenc
```
