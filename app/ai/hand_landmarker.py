import logging
import numpy as np

logger = logging.getLogger(__name__)

# 21 hand landmarks * 3 coordinates = 63-dimensional vector
LANDMARK_DIM = 63


class HandLandmarker:
    """MediaPipe Hands wrapper that extracts and normalises hand landmarks.
    
    Returns a 63-d normalised landmark vector suitable for Qdrant similarity search.
    """

    def __init__(self) -> None:
        self._hands = None
        self.is_loaded: bool = False

    def load(self) -> bool:
        try:
            import mediapipe as mp  # type: ignore
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.5,
            )
            self.is_loaded = True
            logger.info("HandLandmarker loaded")
            return True
        except Exception as exc:
            logger.warning(f"HandLandmarker failed to load: {exc}")
            self.is_loaded = False
            return False

    def unload(self) -> None:
        if self._hands:
            self._hands.close()
        self._hands = None
        self.is_loaded = False

    def extract(self, frame: np.ndarray) -> list[dict]:
        """Return list of hand results, each with normalised vector and handedness."""
        if not self.is_loaded or self._hands is None:
            return []
        try:
            import cv2  # type: ignore
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._hands.process(rgb)
            hands = []
            if results.multi_hand_landmarks:
                for i, lm_list in enumerate(results.multi_hand_landmarks):
                    coords = []
                    for lm in lm_list.landmark:
                        coords.extend([lm.x, lm.y, lm.z])
                    vector = self._normalise(coords)
                    handedness = "RIGHT"
                    if results.multi_handedness and i < len(results.multi_handedness):
                        handedness = results.multi_handedness[i].classification[0].label
                    hands.append({
                        "vector": vector,
                        "handedness": handedness,
                        "raw_landmarks": coords,
                    })
            return hands
        except Exception as exc:
            logger.error(f"Hand landmark extraction error: {exc}")
            return []

    def _normalise(self, coords: list[float]) -> list[float]:
        """Centre landmarks around wrist and normalise to unit range."""
        arr = np.array(coords, dtype=np.float32).reshape(21, 3)
        # Centre on wrist (landmark 0)
        arr = arr - arr[0]
        # Scale
        scale = np.max(np.abs(arr))
        if scale > 0:
            arr = arr / scale
        return arr.flatten().tolist()


hand_landmarker = HandLandmarker()
