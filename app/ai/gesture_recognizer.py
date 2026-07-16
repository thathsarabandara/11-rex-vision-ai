import logging
import numpy as np
from app.config.settings import settings

logger = logging.getLogger(__name__)

# Predefined gesture definitions (used to seed Qdrant on first run)
PREDEFINED_GESTURES = [
    "OPEN_PALM",
    "CLOSED_FIST",
    "THUMBS_UP",
    "THUMBS_DOWN",
    "POINT_LEFT",
    "POINT_RIGHT",
    "STOP_GESTURE",
    "COME_HERE",
]


class GestureRecognizer:
    """Classifies hand gestures by comparing landmark vectors against Qdrant.
    
    For live inference the recogniser receives a 63-d landmark vector and
    queries the gesture collection. Stable detections over N frames are emitted.
    """

    def __init__(self) -> None:
        self._stability_buffer: dict[str, list[str]] = {}  # source_id → recent gesture names

    def evaluate_similarity(
        self,
        query_vector: list[float],
        candidate_vector: list[float],
    ) -> float:
        """Cosine similarity between two landmark vectors."""
        a = np.array(query_vector, dtype=np.float32)
        b = np.array(candidate_vector, dtype=np.float32)
        norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def update_stability(
        self,
        source_id: str,
        gesture_name: str,
        confidence: float,
    ) -> int:
        """Track gesture stability across frames. Returns stable frame count."""
        if confidence < settings.GESTURE_CONFIDENCE_THRESHOLD:
            self._stability_buffer[source_id] = []
            return 0
        buf = self._stability_buffer.setdefault(source_id, [])
        buf.append(gesture_name)
        if len(buf) > settings.GESTURE_STABILITY_FRAMES * 2:
            buf.pop(0)
        return sum(1 for g in buf if g == gesture_name)

    def is_stable(self, source_id: str, gesture_name: str) -> bool:
        buf = self._stability_buffer.get(source_id, [])
        count = sum(1 for g in buf if g == gesture_name)
        return count >= settings.GESTURE_STABILITY_FRAMES

    def clear_source(self, source_id: str) -> None:
        self._stability_buffer.pop(source_id, None)


gesture_recognizer = GestureRecognizer()
