"""
Train YOLOv11 model for CS2 kill feed icon detection.

Prerequisites:
    1. Labeled dataset in YOLO format (from Roboflow export)
    2. data.yaml pointing to train/val images

Usage:
    python train_yolo.py --data dataset/data.yaml
    python train_yolo.py --data dataset/data.yaml --epochs 100 --batch 16
"""

import argparse
import logging
import shutil
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv11 for CS2 kill detection")
    parser.add_argument("--data", required=True,
                        help="Path to data.yaml (Roboflow export)")
    parser.add_argument("--model", default="yolo11n.pt",
                        help="Base model (default: yolo11n.pt)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Training epochs (default: 50)")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size (default: 16)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Image size (default: 640)")
    args = parser.parse_args()

    from ultralytics import YOLO

    logger.info("CS2 Kill Feed Icon - YOLO Training")
    logger.info(f"  Data:   {args.data}")
    logger.info(f"  Model:  {args.model}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch:  {args.batch}")
    logger.info(f"  ImgSz:  {args.imgsz}")

    # Load base model
    model = YOLO(args.model)

    # Train
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        patience=10,         # Early stopping
        save=True,
        plots=True,
    )

    # Copy best weights to models/
    best_pt = Path("runs/detect/train/weights/best.pt")
    if best_pt.exists():
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        dest = models_dir / "best.pt"
        shutil.copy2(best_pt, dest)
        logger.info(f"\nBest model copied to: {dest}")
        logger.info(f"Use with: python main.py --input video.mp4 --model {dest}")
    else:
        logger.info("\nCheck runs/detect/train/weights/ for trained weights")


if __name__ == "__main__":
    main()
