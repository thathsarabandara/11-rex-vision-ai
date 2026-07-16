import logging
import numpy as np
from app.config.settings import settings

logger = logging.getLogger(__name__)


class FaceDetector:
    """Detects faces in a frame using InsightFace.
    
    Returns bounding boxes and face crops for downstream recognition.
    """

    def __init__(self) -> None:
        self._app = None
        self.is_loaded: bool = False
        self.device: str = settings.DEVICE

    def load(self) -> bool:
        try:
            import insightface  # type: ignore
            ctx_id = 0 if self.device == "cuda" else -1
            self._app = insightface.app.FaceAnalysis(
                name=settings.FACE_MODEL_NAME,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"] if ctx_id == 0
                else ["CPUExecutionProvider"],
            )
            self._app.prepare(ctx_id=ctx_id, det_size=(640, 640))
            self.is_loaded = True
            logger.info(f"FaceDetector loaded on ctx_id={ctx_id}")
            return True
        except Exception as exc:
            logger.error(f"FaceDetector failed to load: {exc}")
            self.is_loaded = False
            return False

    def unload(self) -> None:
        self._app = None
        self.is_loaded = False

    def detect(self, frame: np.ndarray) -> list[dict]:
        """Return list of face dicts: bbox, kps, det_score, embedding."""
        if not self.is_loaded or self._app is None:
            return []
        try:
            faces = self._app.get(frame)
            results = []
            for face in faces:
                bbox = face.bbox.astype(int).tolist()
                result: dict = {
                    "bbox": {
                        "x": bbox[0],
                        "y": bbox[1],
                        "width": bbox[2] - bbox[0],
                        "height": bbox[3] - bbox[1],
                    },
                    "det_score": float(face.det_score),
                    "kps": face.kps.tolist() if face.kps is not None else None,
                    "embedding": face.embedding.tolist() if face.embedding is not None else None,
                }
                results.append(result)
            return results
        except Exception as exc:
            logger.error(f"Face detection error: {exc}")
            return []


face_detector = FaceDetector()
