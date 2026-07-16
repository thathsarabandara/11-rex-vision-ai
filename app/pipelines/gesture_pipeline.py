import logging
import numpy as np
from app.ai.hand_landmarker import hand_landmarker
from app.ai.gesture_recognizer import gesture_recognizer
from app.config.settings import settings

logger = logging.getLogger(__name__)


async def run_gesture_pipeline(
    frame: np.ndarray,
    gesture_profiles: list[dict],
) -> list[dict]:
    """Extract hand landmarks, then match against registered gesture profiles."""
    hands = hand_landmarker.extract(frame)
    results = []

    for hand in hands:
        vector = hand.get("vector", [])
        handedness = hand.get("handedness", "RIGHT")

        if not vector or not gesture_profiles:
            continue

        best_score = 0.0
        best_profile = None

        for profile in gesture_profiles:
            profile_vector = profile.get("vector")
            if not profile_vector:
                continue
            score = gesture_recognizer.evaluate_similarity(vector, profile_vector)
            if score > best_score:
                best_score = score
                best_profile = profile

        if best_profile is None or best_score < settings.GESTURE_CONFIDENCE_THRESHOLD:
            continue

        gesture_name = best_profile.get("gesture_name", "UNKNOWN")
        source_id = f"{handedness}"
        stability = gesture_recognizer.update_stability(source_id, gesture_name, best_score)

        results.append({
            "gesture_profile_id": best_profile.get("gesture_profile_id"),
            "gesture_name": gesture_name,
            "confidence": round(best_score, 4),
            "handedness": handedness,
            "stability_frames": stability,
            "action_hint": best_profile.get("action_hint"),
        })

    return results
