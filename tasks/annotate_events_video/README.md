# annotate_events_video

Renders predicted `bounce` and `hit` events onto a video using ASS subtitles.

GPU/NVENC encode:

```bash
docker compose -f compose.yaml -f compose.gpu.yaml run --rm app \
  python tasks/annotate_events_video/run.py \
  --events outputs/detect_events_from_ball/local_table_prior_motion_x30_y10_events.json \
  --video-codec h264_nvenc
```

CPU fallback:

```bash
docker compose run --rm app \
  python tasks/annotate_events_video/run.py \
  --events outputs/detect_events_from_ball/local_table_prior_motion_x30_y10_events.json \
  --video-codec libx264
```

Default input is the local DJI sample video. Generated MP4, ASS, and event JSON outputs should stay under `outputs/`.
