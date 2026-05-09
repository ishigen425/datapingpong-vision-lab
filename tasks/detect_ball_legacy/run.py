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
from datapingpong.tracking import BallBeliefHeatmapTracker, BallHeatmapTracker, HeatmapPeak, extract_top_peaks


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
    parser.add_argument("--decoder", choices=["belief", "track", "argmax"], default="belief")
    parser.add_argument("--candidate-threshold", type=float, default=0.15)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--peak-radius", type=int, default=6)
    parser.add_argument("--track-max-distance", type=float, default=48.0)
    parser.add_argument("--track-distance-weight", type=float, default=0.015)
    parser.add_argument("--track-max-missed", type=int, default=3)
    parser.add_argument("--belief-gravity", type=float, default=0.18)
    parser.add_argument("--belief-prior-blur", type=float, default=3.0)
    parser.add_argument("--belief-measurement-floor", type=float, default=0.02)
    parser.add_argument("--belief-measurement-power", type=float, default=1.5)
    parser.add_argument("--belief-max-missed", type=int, default=100)
    parser.add_argument(
        "--no-belief-reacquire-after-missed",
        action="store_true",
        help="Keep using belief gating after missed frames instead of snapping to a strong new UNet detection.",
    )
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
        decoder=args.decoder,
        candidate_threshold=args.candidate_threshold,
        top_k=args.top_k,
        peak_radius=args.peak_radius,
        track_max_distance=args.track_max_distance,
        track_distance_weight=args.track_distance_weight,
        track_max_missed=args.track_max_missed,
        belief_gravity=args.belief_gravity,
        belief_prior_blur=args.belief_prior_blur,
        belief_measurement_floor=args.belief_measurement_floor,
        belief_measurement_power=args.belief_measurement_power,
        belief_max_missed=args.belief_max_missed,
        belief_reacquire_after_missed=not args.no_belief_reacquire_after_missed,
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
    decoder: str,
    candidate_threshold: float,
    top_k: int,
    peak_radius: int,
    track_max_distance: float,
    track_distance_weight: float,
    track_max_missed: int,
    belief_gravity: float,
    belief_prior_blur: float,
    belief_measurement_floor: float,
    belief_measurement_power: float,
    belief_max_missed: int,
    belief_reacquire_after_missed: bool = True,
) -> list[dict[str, float | int | bool | None]]:
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
    tracker = BallHeatmapTracker(
        max_distance=track_max_distance,
        distance_weight=track_distance_weight,
        max_missed_frames=track_max_missed,
    )
    belief_tracker = BallBeliefHeatmapTracker(
        (360, 640),
        gravity_y=belief_gravity,
        prior_blur_sigma=belief_prior_blur,
        measurement_floor=belief_measurement_floor,
        measurement_power=belief_measurement_power,
        max_missed_frames=belief_max_missed,
        output_threshold=threshold,
        reacquire_after_missed=belief_reacquire_after_missed,
        reacquire_threshold=threshold,
    )

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
            unet_x, unet_y, unet_confidence = raw_heatmap_detection(heatmap, threshold=threshold)
            peaks = extract_top_peaks(
                heatmap,
                threshold=candidate_threshold if decoder in {"belief", "track"} else threshold,
                top_k=top_k if decoder in {"belief", "track"} else 1,
                suppression_radius=peak_radius,
            )
            if decoder == "belief":
                observation = belief_tracker.step(heatmap, peaks=peaks)
                x_value = observation.x
                y_value = observation.y
                confidence = observation.confidence
                candidate_count = observation.candidate_count
            elif decoder == "track":
                observation = tracker.step(peaks)
                x_value = observation.x
                y_value = observation.y
                confidence = observation.confidence
                candidate_count = observation.candidate_count
            else:
                confidence = float(np.max(heatmap))
                if confidence < threshold:
                    x_value = None
                    y_value = None
                    candidate_count = len(peaks)
                else:
                    x_value = peaks[0].x if peaks else None
                    y_value = peaks[0].y if peaks else None
                    candidate_count = len(peaks)
            output_frame = prediction_frame_index(frame_index, window_size=len(frames))

            row = {
                "frame": output_frame,
                "x": x_value,
                "y": y_value,
                "confidence": confidence,
                "candidate_count": candidate_count,
                "unet_x": unet_x,
                "unet_y": unet_y,
                "unet_confidence": unet_confidence,
                "unet_detected": unet_x is not None and unet_y is not None,
            }
            predictions.append(row)
            if debug_stride > 0 and output_frame % debug_stride == 0:
                write_debug_frame(
                    debug_dir,
                    output_frame,
                    frames[len(frames) // 2],
                    x_value,
                    y_value,
                    confidence,
                    unet_x=unet_x,
                    unet_y=unet_y,
                    unet_confidence=unet_confidence,
                    peaks=peaks,
                )

    capture.release()

    payload = {
        "video": str(video_path),
        "input_size": {"width": 640, "height": 360},
        "frames_read": frame_index + 1,
        "start_frame": start_frame,
        "max_frames": max_frames,
        "debug_stride": debug_stride,
        "window_size": 9,
        "decoder": decoder,
        "candidate_threshold": candidate_threshold,
        "top_k": top_k,
        "peak_radius": peak_radius,
        "track_max_distance": track_max_distance,
        "track_distance_weight": track_distance_weight,
        "track_max_missed": track_max_missed,
        "belief_gravity": belief_gravity,
        "belief_prior_blur": belief_prior_blur,
        "belief_measurement_floor": belief_measurement_floor,
        "belief_measurement_power": belief_measurement_power,
        "belief_max_missed": belief_max_missed,
        "belief_reacquire_after_missed": belief_reacquire_after_missed,
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


def prediction_frame_index(current_frame: int, *, window_size: int) -> int:
    return current_frame - window_size // 2


def raw_heatmap_detection(heatmap: np.ndarray, *, threshold: float) -> tuple[int | None, int | None, float]:
    confidence = float(np.max(heatmap)) if heatmap.size else 0.0
    if confidence < threshold:
        return None, None, confidence
    y, x = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
    return int(x), int(y), confidence


def write_debug_frame(
    debug_dir: Path,
    frame_index: int,
    rgb_frame: np.ndarray,
    x: int | None,
    y: int | None,
    confidence: float,
    *,
    unet_x: int | None = None,
    unet_y: int | None = None,
    unet_confidence: float = 0.0,
    peaks: list[HeatmapPeak],
) -> None:
    image = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    for peak in peaks:
        cv2.circle(image, (peak.x, peak.y), 3, (255, 180, 0), 1, cv2.LINE_AA)
    if unet_x is not None and unet_y is not None:
        cv2.drawMarker(image, (unet_x, unet_y), (255, 0, 255), cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
    if x is not None and y is not None:
        cv2.circle(image, (x, y), 5, (0, 220, 60), 2)
    cv2.putText(
        image,
        f"frame={frame_index} track={confidence:.3f} unet={unet_confidence:.3f}",
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
