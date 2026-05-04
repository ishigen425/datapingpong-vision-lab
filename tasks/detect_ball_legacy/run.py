from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from datapingpong.models.legacy_ball_tracking import UNet


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the legacy ball-tracking UNet on a video.")
    parser.add_argument("--video", type=Path, default=ROOT / "data/raw/DJI_0056_001-001.MP4")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "models/checkpoints/legacy/ball_tracking/20211107/epoch_27_4030.zip",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/detect_ball_legacy/predictions.json")
    parser.add_argument("--debug-dir", type=Path, default=ROOT / "outputs/detect_ball_legacy/debug_frames")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=20)
    parser.add_argument("--debug-stride", type=int, default=1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    device = select_device(args.device)
    model = load_model(args.checkpoint, device)
    predictions = run_video(
        model=model,
        video_path=args.video,
        output_path=args.output,
        debug_dir=args.debug_dir,
        start_frame=args.start_frame,
        max_frames=args.max_frames,
        debug_stride=args.debug_stride,
        device=device,
        threshold=args.threshold,
    )

    print(f"device: {device}")
    print(f"video: {args.video}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"predictions: {args.output}")
    print(f"debug_frames: {args.debug_dir}")
    print(f"predicted_rows: {len(predictions)}")
    return 0


def select_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint_path: Path, device: torch.device) -> UNet:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model = UNet(27).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def run_video(
    model: UNet,
    video_path: Path,
    output_path: Path,
    debug_dir: Path,
    start_frame: int,
    max_frames: int,
    debug_stride: int,
    device: torch.device,
    threshold: float,
) -> list[dict[str, float | int | None]]:
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    if start_frame > 0:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    frames: list[np.ndarray] = []
    predictions: list[dict[str, float | int | None]] = []
    frame_index = start_frame - 1
    started = perf_counter()

    with torch.no_grad():
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_index += 1
            if max_frames > 0 and frame_index >= start_frame + max_frames:
                break

            image = preprocess_frame(frame)
            frames.append(image)
            if len(frames) < 9:
                continue
            if len(frames) > 9:
                frames = frames[-9:]

            tensor = stack_frames(frames).to(device)
            heatmap = model(tensor).squeeze().detach().cpu().numpy()
            confidence = float(np.max(heatmap))
            y, x = np.unravel_index(int(np.argmax(np.where(heatmap < threshold, 0.0, heatmap))), heatmap.shape)
            if confidence < threshold:
                x_value: int | None = None
                y_value: int | None = None
            else:
                x_value = int(x)
                y_value = int(y)

            row = {
                "frame": frame_index,
                "x": x_value,
                "y": y_value,
                "confidence": confidence,
            }
            predictions.append(row)
            if debug_stride > 0 and frame_index % debug_stride == 0:
                write_debug_frame(debug_dir, frame_index, frames[4], x_value, y_value, confidence)

    capture.release()

    payload = {
        "video": str(video_path),
        "input_size": {"width": 640, "height": 360},
        "frames_read": frame_index + 1,
        "start_frame": start_frame,
        "max_frames": max_frames,
        "debug_stride": debug_stride,
        "window_size": 9,
        "elapsed_s": round(perf_counter() - started, 3),
        "predictions": predictions,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return predictions


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(frame, (3, 3), 0)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(blurred, kernel, iterations=1)
    rgb = cv2.cvtColor(dilated, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (640, 360))


def stack_frames(frames: list[np.ndarray]) -> torch.Tensor:
    stacked = np.dstack(frames).transpose((2, 0, 1))
    return torch.from_numpy(stacked[np.newaxis, :] / 255.0).float()


def write_debug_frame(
    debug_dir: Path,
    frame_index: int,
    rgb_frame: np.ndarray,
    x: int | None,
    y: int | None,
    confidence: float,
) -> None:
    image = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    if x is not None and y is not None:
        cv2.circle(image, (x, y), 4, (0, 120, 243), 2)
    cv2.putText(
        image,
        f"frame={frame_index} conf={confidence:.3f}",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(debug_dir / f"{frame_index:05d}.jpg"), image)


if __name__ == "__main__":
    raise SystemExit(main())
