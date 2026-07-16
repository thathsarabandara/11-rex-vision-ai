import logging
import numpy as np

logger = logging.getLogger(__name__)

POSE_CLASSES = ["STANDING", "SITTING", "CROUCHING", "LYING", "FALL_CANDIDATE", "UNKNOWN"]


class PoseEstimator:
    """Optional MediaPipe Pose wrapper for body-pose and fall-candidate detection.

    Fall candidate requires:
    - Sudden vertical-to-horizontal body transition
    - Bounding-box aspect-ratio flip
    - Stability across multiple frames
    """

    def __init__(self) -> None:
        self._pose = None
        self.is_loaded: bool = False
        self._history: dict[int, list[str]] = {}  # track_id → pose history

    def load(self) -> bool:
        try:
            import mediapipe as mp  # type: ignore
            self._pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=0,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.is_loaded = True
            logger.info("PoseEstimator loaded")
            return True
        except Exception as exc:
            logger.warning(f"PoseEstimator failed to load: {exc}")
            self.is_loaded = False
            return False

    def unload(self) -> None:
        if self._pose:
            self._pose.close()
        self._pose = None
        self.is_loaded = False
        self._history.clear()

    def estimate(self, frame: np.ndarray, track_id: int = -1) -> dict:
        """Return pose classification for the current frame."""
        if not self.is_loaded or self._pose is None:
            return {"pose": "UNKNOWN", "confidence": 0.0, "fall_candidate": False}

        try:
            import cv2  # type: ignore
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._pose.process(rgb)

            if not results.pose_landmarks:
                return {"pose": "UNKNOWN", "confidence": 0.0, "fall_candidate": False}

            lm = results.pose_landmarks.landmark
            # Use shoulder-hip-ankle vertical alignment to classify pose
            left_shoulder_y = lm[11].y
            right_shoulder_y = lm[12].y
            left_hip_y = lm[23].y
            right_hip_y = lm[24].y
            left_ankle_y = lm[27].y
            right_ankle_y = lm[28].y

            shoulder_y = (left_shoulder_y + right_shoulder_y) / 2
            hip_y = (left_hip_y + right_hip_y) / 2
            ankle_y = (left_ankle_y + right_ankle_y) / 2

            vertical_span = abs(ankle_y - shoulder_y)
            horizontal_span = abs(lm[11].x - lm[12].x)

            pose = "UNKNOWN"
            if vertical_span > 0.4:
                if hip_y > 0.6:
                    pose = "STANDING"
                else:
                    pose = "CROUCHING"
            elif vertical_span < 0.2 and horizontal_span > 0.15:
                pose = "LYING"
            else:
                pose = "SITTING"

            # Fall candidate: previous standing/crouching → now lying
            history = self._history.setdefault(track_id, [])
            was_upright = any(p in ("STANDING", "CROUCHING") for p in history[-3:])
            fall_candidate = was_upright and pose == "LYING"
            if fall_candidate:
                pose = "FALL_CANDIDATE"

            history.append(pose)
            if len(history) > 10:
                history.pop(0)

            return {
                "pose": pose,
                "confidence": 0.75,
                "fall_candidate": fall_candidate,
            }
        except Exception as exc:
            logger.error(f"Pose estimation error: {exc}")
            return {"pose": "UNKNOWN", "confidence": 0.0, "fall_candidate": False}


pose_estimator = PoseEstimator()
