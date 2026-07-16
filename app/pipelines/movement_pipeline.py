import logging
from app.ai.tracker import tracker

logger = logging.getLogger(__name__)

MOVEMENT_VELOCITY_THRESHOLD = 0.05
SCALE_APPROACH_THRESHOLD = 0.02


def classify_movement(
    prev_cx: float, curr_cx: float,
    prev_cy: float, curr_cy: float,
    prev_area: float, curr_area: float,
) -> tuple[str, float, float, float]:
    """Classify movement direction and scale change from bounding-box trajectory."""
    vx = curr_cx - prev_cx
    vy = curr_cy - prev_cy
    scale_change = (curr_area - prev_area) / (prev_area + 1e-6)

    if abs(vx) < MOVEMENT_VELOCITY_THRESHOLD and abs(vy) < MOVEMENT_VELOCITY_THRESHOLD:
        if abs(scale_change) < SCALE_APPROACH_THRESHOLD:
            return "STATIONARY", vx, vy, scale_change

    if abs(scale_change) >= SCALE_APPROACH_THRESHOLD:
        if scale_change > 0:
            return "MOVING_TOWARD_CAMERA", vx, vy, scale_change
        else:
            return "MOVING_AWAY_FROM_CAMERA", vx, vy, scale_change

    if abs(vx) >= abs(vy):
        return ("MOVING_LEFT" if vx < 0 else "MOVING_RIGHT"), vx, vy, scale_change

    return "UNKNOWN", vx, vy, scale_change


async def run_movement_pipeline(detections: list[dict]) -> list[dict]:
    """Analyse bounding-box trajectory to estimate person movement."""
    results = []

    for det in detections:
        if det.get("class_name") != "person":
            continue
        track_id = det.get("track_id")
        if track_id is None:
            continue

        history = tracker.get_track_history(track_id)
        if len(history) < 3:
            continue

        prev = history[-3]
        curr = history[-1]

        prev_cx = prev["x"] + prev["width"] / 2
        prev_cy = prev["y"] + prev["height"] / 2
        curr_cx = curr["x"] + curr["width"] / 2
        curr_cy = curr["y"] + curr["height"] / 2
        prev_area = prev["width"] * prev["height"]
        curr_area = curr["width"] * curr["height"]

        movement, vx, vy, scale = classify_movement(
            prev_cx, curr_cx, prev_cy, curr_cy, prev_area, curr_area
        )

        results.append({
            "track_id": track_id,
            "movement": movement,
            "velocity_x": round(float(vx), 4),
            "velocity_y": round(float(vy), 4),
            "scale_change": round(float(scale), 4),
            "confidence": 0.80,
        })

    return results
