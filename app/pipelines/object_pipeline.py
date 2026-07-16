import numpy as np
from app.ai.object_detector import object_detector
from app.ai.tracker import tracker


async def run_object_pipeline(frame: np.ndarray, features: dict) -> list[dict]:
    """Run YOLO detection and update tracker. Returns enriched track list."""
    raw_detections = object_detector.detect(frame)
    tracked = tracker.update(raw_detections)
    return tracked
