from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any
import urllib.request

import cv2


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from datapingpong.pose import MultiPoseFrame, PoseInstance, PoseLandmark, assign_pose_roles, landmark_names, multi_pose_frames_to_dicts


MODEL_URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate player pose landmarks from a video with MediaPipe Pose.")
    parser.add_argument("--video", type=Path, default=ROOT / "data/raw/DJI_0056_001-001.MP4")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/estimate_pose/DJI_0056_001_pose.json")
    parser.add_argument("--model", type=Path, default=ROOT / "models/checkpoints/mediapipe/pose_landmarker_full.task")
    parser.add_argument("--num-poses", type=int, default=2)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0, help="Process at most this many frames. Use 0 for all frames.")
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-presence-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    args = parser.parse_args()

    video_info = read_video_info(args.video)
    model_downloaded = ensure_model_asset(args.model)
    started = perf_counter()
    frames = run_pose_estimation(
        args.video,
        model_path=args.model,
        fps=video_info["fps"],
        num_poses=args.num_poses,
        start_frame=args.start_frame,
        max_frames=args.max_frames,
        min_detection_confidence=args.min_detection_confidence,
        min_presence_confidence=args.min_presence_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "task": "estimate_pose",
        "pose_backend": "mediapipe_pose",
        "video": str(args.video),
        "model": str(args.model),
        "num_poses": args.num_poses,
        "fps": video_info["fps"],
        "frame_width": video_info["width"],
        "frame_height": video_info["height"],
        "frame_count": video_info["frames"],
        "start_frame": args.start_frame,
        "max_frames": args.max_frames,
        "min_detection_confidence": args.min_detection_confidence,
        "min_presence_confidence": args.min_presence_confidence,
        "min_tracking_confidence": args.min_tracking_confidence,
        "model_downloaded": model_downloaded,
        "landmark_names": landmark_names(),
        "elapsed_s": round(perf_counter() - started, 3),
        "frames": multi_pose_frames_to_dicts(frames),
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    detected_frames = sum(1 for frame in frames if frame.detected)
    multi_pose_frames = sum(1 for frame in frames if len(frame.poses) >= 2)
    print(
        json.dumps(
            {
                "video": str(args.video),
                "output": str(args.output),
                "model": str(args.model),
                "processed_frames": len(frames),
                "detected_frames": detected_frames,
                "multi_pose_frames": multi_pose_frames,
                "coverage": round(detected_frames / len(frames), 4) if frames else 0.0,
                "pose_backend": "mediapipe_pose",
            },
            indent=2,
        )
    )
    return 0


def run_pose_estimation(
    video_path: Path,
    *,
    model_path: Path,
    fps: float,
    num_poses: int,
    start_frame: int,
    max_frames: int,
    min_detection_confidence: float,
    min_presence_confidence: float,
    min_tracking_confidence: float,
) -> list[MultiPoseFrame]:
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    from mediapipe import Image, ImageFormat
    from mediapipe.tasks.python.core.base_options import BaseOptions
    from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
    from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarker, PoseLandmarkerOptions

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    if start_frame > 0:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames: list[PoseFrame] = []
    frame_index = start_frame - 1
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=VisionTaskRunningMode.VIDEO,
        num_poses=num_poses,
        min_pose_detection_confidence=min_detection_confidence,
        min_pose_presence_confidence=min_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
        output_segmentation_masks=False,
    )
    with PoseLandmarker.create_from_options(options) as estimator:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_index += 1
            if max_frames > 0 and len(frames) >= max_frames:
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp_ms = int(round(frame_index * 1000.0 / fps)) if fps > 0.0 else frame_index
            result = estimator.detect_for_video(Image(image_format=ImageFormat.SRGB, data=rgb_frame), timestamp_ms)
            poses = assign_pose_roles(
                [
                    PoseInstance(
                        role=f"pose_{pose_index}",
                        landmarks=_landmark_mapping(pose_landmarks),
                        world_landmarks=_landmark_mapping(
                            result.pose_world_landmarks[pose_index] if pose_index < len(result.pose_world_landmarks) else []
                        ),
                    )
                    for pose_index, pose_landmarks in enumerate(result.pose_landmarks)
                ]
            )
            timestamp_ms = round(frame_index * 1000.0 / fps, 3) if fps > 0.0 else None
            frames.append(MultiPoseFrame(frame=frame_index, timestamp_ms=timestamp_ms, poses=poses))
    capture.release()
    return frames


def read_video_info(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if fps <= 0.0:
        raise ValueError(f"Could not read FPS from {path}")
    return {"fps": fps, "width": width, "height": height, "frames": frames}


def _landmark_mapping(landmarks: list[Any]) -> dict[str, PoseLandmark]:
    return {
        name: PoseLandmark(
            x=float(landmark.x),
            y=float(landmark.y),
            z=float(landmark.z),
            visibility=_optional_float(landmark, "visibility"),
            presence=_optional_float(landmark, "presence"),
        )
        for name, landmark in zip(landmark_names(), landmarks, strict=False)
    }


def _optional_float(landmark: Any, field: str) -> float | None:
    value = getattr(landmark, field, None)
    return float(value) if value is not None else None


def ensure_model_asset(path: Path) -> bool:
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(resolve_model_url(path)) as response:
        path.write_bytes(response.read())
    return True


def resolve_model_url(path: Path) -> str:
    name = path.name.lower()
    for variant, url in MODEL_URLS.items():
        if variant in name:
            return url
    return MODEL_URLS["full"]


if __name__ == "__main__":
    raise SystemExit(main())
