"""
Extract frames from CS2 gameplay videos for YOLO training data.

Extracts frames at specified FPS and saves as images.
Then label them on https://app.roboflow.com (free) or with LabelImg.

Usage:
    python extract_frames.py --input downloads/video.mp4
    python extract_frames.py --input downloads/ --fps 2
"""

import argparse
import logging
from pathlib import Path

import cv2

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def extract_frames(video_path, output_dir, fps=2):
    """Extract frames from video at given FPS."""
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Cannot open: {video_path}")
        return 0

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps if video_fps > 0 else 0

    logger.info(f"Video: {video_path.name} ({duration:.1f}s, {video_fps:.0f}fps)")

    frame_interval = max(1, int(video_fps / fps))
    saved = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            timestamp = frame_idx / video_fps
            filename = f"{video_path.stem}_f{frame_idx:05d}_{timestamp:.2f}s.jpg"
            cv2.imwrite(str(output_dir / filename), frame)
            saved += 1

        frame_idx += 1

    cap.release()
    logger.info(f"  Saved {saved} frames to {output_dir}/")
    return saved


def main():
    parser = argparse.ArgumentParser(description="Extract frames for YOLO training")
    parser.add_argument("--input", "-i", required=True,
                        help="Video file or folder")
    parser.add_argument("--output", "-o", default="training_frames",
                        help="Output directory (default: training_frames)")
    parser.add_argument("--fps", type=float, default=2,
                        help="Frames per second to extract (default: 2)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.is_file():
        videos = [input_path]
    elif input_path.is_dir():
        videos = sorted(input_path.glob("*.mp4"))
    else:
        logger.error(f"Not found: {input_path}")
        return

    total = 0
    for video in videos:
        total += extract_frames(video, args.output, args.fps)

    logger.info(f"\nTotal: {total} frames from {len(videos)} videos")
    logger.info(f"Output: {args.output}/")
    logger.info(f"\nNext steps:")
    logger.info(f"  1. Go to https://app.roboflow.com (free)")
    logger.info(f"  2. Create project -> Object Detection")
    logger.info(f"  3. Upload frames from {args.output}/")
    logger.info(f"  4. Label 'kill_feed_icon' on kill feed entries")
    logger.info(f"  5. Export as 'YOLOv11' format")
    logger.info(f"  6. Train: python train_yolo.py --data dataset/data.yaml")


if __name__ == "__main__":
    main()
