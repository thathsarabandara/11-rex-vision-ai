"""Evaluate a trained YOLO object detection model.

Run directly:
    python training/object_detection/evaluate.py \\
        --model runs/train/weights/best.pt \\
        --data path/to/data.yaml
"""
import argparse
import json
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate YOLO object detection model")
    p.add_argument("--model", required=True, help="Path to trained .pt model")
    p.add_argument("--data", required=True, help="Path to data.yaml")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0")
    return p.parse_args()


def main():
    args = parse_args()
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError:
        logger.error("ultralytics not installed")
        sys.exit(1)

    model = YOLO(args.model)
    metrics = model.val(data=args.data, imgsz=args.imgsz, device=args.device)

    results = {
        "mAP50": round(float(metrics.box.map50), 4),
        "mAP50_95": round(float(metrics.box.map), 4),
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
    }
    print(json.dumps(results, indent=2))
    logger.info(f"Evaluation complete: {results}")


if __name__ == "__main__":
    main()
