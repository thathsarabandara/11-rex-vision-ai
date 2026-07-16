import logging
import numpy as np
from app.ai.expression_estimator import expression_estimator

logger = logging.getLogger(__name__)


async def run_expression_pipeline(frame: np.ndarray, face_results: list[dict]) -> list[dict]:
    """Run expression estimation on each detected face crop."""
    results = []
    for i, face in enumerate(face_results):
        bbox = face.get("bbox", {})
        x, y, w, h = bbox.get("x", 0), bbox.get("y", 0), bbox.get("width", 0), bbox.get("height", 0)

        if w <= 0 or h <= 0:
            continue

        # Extract face crop (with small padding)
        pad = 10
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame.shape[1], x + w + pad)
        y2 = min(frame.shape[0], y + h + pad)
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        est = expression_estimator.estimate(crop, track_id=i)
        results.append({
            "face_index": i,
            "face_track_id": i,
            "expression": est["expression"],
            "confidence": est["confidence"],
            "stability_frames": est["stability_frames"],
        })

    return results
