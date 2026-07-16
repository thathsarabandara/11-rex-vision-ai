import logging
import numpy as np
from app.config.settings import settings

logger = logging.getLogger(__name__)

EXPRESSION_CLASSES = ["NEUTRAL", "HAPPY", "SAD", "ANGRY", "SURPRISED", "FEARFUL", "DISGUSTED"]


class ExpressionEstimator:
    """Lightweight facial-expression estimation using an ONNX model.
    
    IMPORTANT: Facial-expression estimation does not reliably determine a
    person's true emotional state. Results are probabilistic visual estimates only.
    """

    def __init__(self) -> None:
        self._session = None
        self.is_loaded: bool = False
        self._input_name: str = ""
        self._stability_buffer: dict[int, list[str]] = {}  # track_id → recent predictions

    def load(self) -> bool:
        try:
            import onnxruntime as ort  # type: ignore
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if settings.DEVICE == "cuda" \
                else ["CPUExecutionProvider"]
            self._session = ort.InferenceSession(settings.EXPRESSION_MODEL_PATH, providers=providers)
            self._input_name = self._session.get_inputs()[0].name
            self.is_loaded = True
            logger.info("ExpressionEstimator loaded")
            return True
        except Exception as exc:
            logger.warning(f"ExpressionEstimator failed to load: {exc}")
            self.is_loaded = False
            return False

    def unload(self) -> None:
        self._session = None
        self.is_loaded = False
        self._stability_buffer.clear()

    def estimate(self, face_crop: np.ndarray, track_id: int = -1) -> dict:
        """Estimate expression for a single face crop.
        
        Returns expression label and confidence. Returns UNCERTAIN if model unavailable
        or confidence is below threshold.
        """
        if not self.is_loaded or self._session is None:
            return {"expression": "UNCERTAIN", "confidence": 0.0, "stability_frames": 0}

        try:
            # Preprocess: resize to 48x48, normalise
            import cv2  # type: ignore
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (48, 48)).astype(np.float32) / 255.0
            inp = resized[np.newaxis, np.newaxis, :, :]  # (1, 1, 48, 48)

            outputs = self._session.run(None, {self._input_name: inp})
            probs = outputs[0][0]
            idx = int(np.argmax(probs))
            confidence = float(probs[idx])

            if confidence < settings.EXPRESSION_CONFIDENCE_THRESHOLD:
                expression = "UNCERTAIN"
            else:
                expression = EXPRESSION_CLASSES[idx] if idx < len(EXPRESSION_CLASSES) else "UNCERTAIN"

            # Stability tracking
            buf = self._stability_buffer.setdefault(track_id, [])
            buf.append(expression)
            if len(buf) > 10:
                buf.pop(0)
            stability = sum(1 for e in buf if e == expression)

            return {
                "expression": expression,
                "confidence": round(confidence, 4),
                "stability_frames": stability,
            }
        except Exception as exc:
            logger.error(f"Expression estimation error: {exc}")
            return {"expression": "UNCERTAIN", "confidence": 0.0, "stability_frames": 0}

    def clear_track(self, track_id: int) -> None:
        self._stability_buffer.pop(track_id, None)


expression_estimator = ExpressionEstimator()
