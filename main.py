"""
CS2 Auto Clipper - YOLO-based kill detection pipeline.

Detects kill moments in CS2 gameplay videos using a custom-trained
YOLOv11 model that recognizes kill feed icons, then clips them with ffmpeg.

Usage:
    python main.py --input video.mp4
    python main.py --input clips_folder/
    python main.py --input video.mp4 --model runs/train/weights/best.pt
"""

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

import cv2

# ============================================================
# YOLO MODEL YOLU - Kendi eğittiğin modeli buraya koy:
# Eğitimden sonra: runs/detect/train/weights/best.pt
# ============================================================
DEFAULT_MODEL_PATH = "models/best.pt"

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class KillDetector:
    """YOLO-based kill feed icon detector."""

    def __init__(self, model_path=DEFAULT_MODEL_PATH, confidence=0.75, sample_fps=3, cooldown=3.0):
        """
        Args:
            model_path: Path to YOLO .pt weights file.
            confidence: Minimum detection confidence (0-1).
            sample_fps: Frames to process per second (3-4 optimal).
            cooldown: Seconds to skip after a kill detection.
        """
        self.confidence = confidence
        self.sample_fps = sample_fps
        self.cooldown = cooldown

        logger.info(f"Loading YOLO model: {model_path}")
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            logger.info(f"Model loaded (conf={confidence}, sample_fps={sample_fps}, cooldown={cooldown}s)")
        except FileNotFoundError:
            logger.error(f"Model not found: {model_path}")
            logger.error("Train a model first or provide --model path")
            sys.exit(1)

    def detect(self, video_path):
        """
        Detect kill moments in a video.

        Returns list of dicts:
            [{"timestamp": float, "frame": int, "confidence": float}, ...]
        """
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            logger.error(f"Cannot open: {video_path}")
            return []

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps if video_fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(f"Video: {video_path.name} ({width}x{height}, {video_fps:.0f}fps, {duration:.1f}s)")

        # Her N. kareyi oku (sample_fps'e göre)
        frame_interval = max(1, int(video_fps / self.sample_fps))
        kills = []
        last_kill_time = -self.cooldown  # İlk kill'e izin ver
        frame_idx = 0
        processed = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                timestamp = frame_idx / video_fps
                processed += 1

                # Cooldown kontrolü - son kill'den beri yeterli süre geçti mi?
                if (timestamp - last_kill_time) < self.cooldown:
                    frame_idx += 1
                    continue

                # YOLO inference
                results = self.model(frame, conf=self.confidence, verbose=False)

                for r in results:
                    if r.boxes is not None and len(r.boxes) > 0:
                        # En yüksek confidence'lı detection'ı al
                        best_conf = float(r.boxes.conf.max())
                        kills.append({
                            "timestamp": round(timestamp, 2),
                            "frame": frame_idx,
                            "confidence": round(best_conf, 4),
                        })
                        last_kill_time = timestamp
                        logger.info(f"  Kill detected: {timestamp:.2f}s (conf={best_conf:.3f})")
                        break  # Bu frame'de 1 kill yeter

            frame_idx += 1

        cap.release()
        logger.info(f"Processed {processed} frames, found {len(kills)} kills")
        return kills


class VideoClipper:
    """FFmpeg-based lossless video clipper."""

    def __init__(self, pre_seconds=3.0, post_seconds=2.0, output_dir="clips"):
        """
        Args:
            pre_seconds: Seconds to include before kill moment.
            post_seconds: Seconds to include after kill moment.
            output_dir: Directory to save clips.
        """
        self.pre_seconds = pre_seconds
        self.post_seconds = post_seconds
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def clip(self, video_path, kills):
        """
        Cut clips around kill moments using ffmpeg -c copy (lossless).

        Returns list of output clip paths.
        """
        video_path = Path(video_path)
        clip_paths = []

        if not kills:
            logger.info("No kills to clip")
            return clip_paths

        for i, kill in enumerate(kills, 1):
            ts = kill["timestamp"]
            start = max(0, ts - self.pre_seconds)
            duration = self.pre_seconds + self.post_seconds

            output_name = f"{video_path.stem}_kill{i}_{ts:.1f}s.mp4"
            output_path = self.output_dir / output_name

            cmd = [
                "ffmpeg", "-y",
                "-ss", f"{start:.3f}",
                "-i", str(video_path),
                "-t", f"{duration:.3f}",
                "-c", "copy",           # Sıfır kalite kaybı
                "-avoid_negative_ts", "make_zero",
                str(output_path),
            ]

            try:
                subprocess.run(cmd, capture_output=True, check=True, timeout=30)
                clip_paths.append(output_path)
                logger.info(f"  Clip {i}/{len(kills)}: {output_name} ({start:.1f}s - {start+duration:.1f}s)")
            except subprocess.CalledProcessError as e:
                logger.error(f"  FFmpeg failed for kill at {ts:.2f}s: {e.stderr[:200] if e.stderr else ''}")
            except subprocess.TimeoutExpired:
                logger.error(f"  FFmpeg timeout for kill at {ts:.2f}s")

        logger.info(f"Created {len(clip_paths)}/{len(kills)} clips in {self.output_dir}/")
        return clip_paths


def process_video(video_path, detector, clipper):
    """Process a single video: detect kills + clip."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing: {video_path}")
    logger.info(f"{'='*60}")

    start_time = time.time()

    # 1. Detect kills
    kills = detector.detect(video_path)

    if not kills:
        logger.info("No kills found, skipping clip extraction")
        return []

    # 2. Clip
    clips = clipper.clip(video_path, kills)

    elapsed = time.time() - start_time
    logger.info(f"Done: {len(kills)} kills, {len(clips)} clips ({elapsed:.1f}s)")

    return clips


def main():
    parser = argparse.ArgumentParser(description="CS2 Auto Clipper - YOLO Kill Detection")
    parser.add_argument("--input", "-i", required=True,
                        help="Video file or folder containing .mp4 files")
    # ============================================================
    # MODEL YOLU: Eğitimden sonra best.pt yolunu buraya ver
    # Örnek: --model runs/detect/train/weights/best.pt
    # ============================================================
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL_PATH,
                        help=f"YOLO model path (default: {DEFAULT_MODEL_PATH})")
    parser.add_argument("--confidence", "-c", type=float, default=0.75,
                        help="Min detection confidence (default: 0.75)")
    parser.add_argument("--sample-fps", type=int, default=3,
                        help="Frames to process per second (default: 3)")
    parser.add_argument("--cooldown", type=float, default=3.0,
                        help="Seconds between kill detections (default: 3.0)")
    parser.add_argument("--pre", type=float, default=3.0,
                        help="Seconds before kill to include in clip (default: 3.0)")
    parser.add_argument("--post", type=float, default=2.0,
                        help="Seconds after kill to include in clip (default: 2.0)")
    parser.add_argument("--output", "-o", default="clips",
                        help="Output directory for clips (default: clips)")
    args = parser.parse_args()

    logger.info("CS2 Auto Clipper - YOLO Kill Detection")
    logger.info(f"Model: {args.model}")
    logger.info(f"Confidence: {args.confidence}, Sample FPS: {args.sample_fps}, Cooldown: {args.cooldown}s")

    # Init
    detector = KillDetector(
        model_path=args.model,
        confidence=args.confidence,
        sample_fps=args.sample_fps,
        cooldown=args.cooldown,
    )
    clipper = VideoClipper(
        pre_seconds=args.pre,
        post_seconds=args.post,
        output_dir=args.output,
    )

    # Find videos
    input_path = Path(args.input)
    if input_path.is_file():
        videos = [input_path]
    elif input_path.is_dir():
        videos = sorted(input_path.glob("*.mp4"))
    else:
        logger.error(f"Not found: {input_path}")
        sys.exit(1)

    if not videos:
        logger.error(f"No .mp4 files found in {input_path}")
        sys.exit(1)

    logger.info(f"Found {len(videos)} video(s)")

    # Process
    total_kills = 0
    total_clips = 0
    for video in videos:
        clips = process_video(str(video), detector, clipper)
        total_kills += len(clips)
        total_clips += len(clips)

    logger.info(f"\n{'='*60}")
    logger.info(f"All done: {len(videos)} videos, {total_kills} kills, {total_clips} clips")
    logger.info(f"Clips saved to: {args.output}/")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
