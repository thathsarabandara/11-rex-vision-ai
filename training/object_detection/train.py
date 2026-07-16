"""Object detection fine-tuning script using Ultralytics YOLO.

Run directly:
    python training/object_detection/train.py \\
        --data path/to/data.yaml \\
        --model yolov8s.pt \\
        --epochs 50 \\
        --imgsz 640

Results are saved to runs/ and model artifacts to the path specified.
"""
import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune YOLO for REX object detection")
    p.add_argument("--data", required=True, help="Path to data.yaml")
    p.add_argument("--model", default="yolov8s.pt", help="Base YOLO model")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="0", help="cuda device or 'cpu'")
    p.add_argument("--output", default="runs/train", help="Output directory")
    return p.parse_args()


def main():
    args = parse_args()
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError:
        logger.error("ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    logger.info(f"Loading base model: {args.model}")
    model = YOLO(args.model)

    logger.info(f"Starting training: epochs={args.epochs} imgsz={args.imgsz} device={args.device}")
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.output,
    )

    logger.info("Training complete.")
    logger.info(f"Best model saved to: {results.save_dir}/weights/best.pt")


if __name__ == "__main__":
    main()
