import logging
import time
from typing import Optional
import numpy as np
from app.config.settings import settings

logger = logging.getLogger(__name__)


class ObjectDetector:
    """Wraps Ultralytics YOLO for object detection.
    
    Lazily loads the model on first use. Falls back gracefully if model file
    is missing or CUDA is unavailable.
    """

    # COCO classes of interest
    ENABLED_CLASSES = {
        0: "person", 15: "cat", 16: "dog",
        56: "chair", 60: "table", 39: "bottle",
        41: "cup", 24: "backpack", 63: "laptop",
        67: "cell phone", 72: "tv",
    }

    def __init__(self) -> None:
        self._model = None
        self.is_loaded: bool = False
        self.device: str = settings.DEVICE
        self._last_inference_ms: float = 0.0

    def load(self) -> bool:
        try:
            from ultralytics import YOLO  # type: ignore
            self._model = YOLO(settings.OBJECT_MODEL_PATH)
            self._model.to(self.device)
            self.is_loaded = True
            logger.info(f"ObjectDetector loaded on {self.device}")
            return True
        except Exception as exc:
            logger.warning(f"ObjectDetector failed to load: {exc}. Falling back to CPU or stub.")
            if self.device != "cpu":
                try:
                    from ultralytics import YOLO  # type: ignore
                    self._model = YOLO(settings.OBJECT_MODEL_PATH)
                    self._model.to("cpu")
                    self.device = "cpu"
                    self.is_loaded = True
                    logger.info("ObjectDetector loaded on CPU fallback")
                    return True
                except Exception as exc2:
                    logger.error(f"ObjectDetector CPU fallback also failed: {exc2}")
            self.is_loaded = False
            return False

    def unload(self) -> None:
        self._model = None
        self.is_loaded = False
        logger.info("ObjectDetector unloaded")

    def detect(self, frame: np.ndarray) -> list[dict]:
        if not self.is_loaded or self._model is None:
            return []
        try:
            t0 = time.monotonic()
            results = self._model(
                frame,
                conf=settings.OBJECT_CONFIDENCE_THRESHOLD,
                iou=settings.OBJECT_IOU_THRESHOLD,
                imgsz=(settings.VISION_INPUT_HEIGHT, settings.VISION_INPUT_WIDTH),
                verbose=False,
            )
            self._last_inference_ms = (time.monotonic() - t0) * 1000

            detections = []
            for r in results:
                boxes = r.boxes
                if boxes is None:
                    continue
                for box in boxes:
                    cls_id = int(box.cls[0])
                    cls_name = self.ENABLED_CLASSES.get(cls_id, r.names.get(cls_id, str(cls_id)))
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    detections.append({
                        "class_id": cls_id,
                        "class_name": cls_name,
                        "confidence": round(float(box.conf[0]), 4),
                        "bounding_box": {
                            "x": x1,
                            "y": y1,
                            "width": x2 - x1,
                            "height": y2 - y1,
                        },
                        "track_id": int(box.id[0]) if box.id is not None else None,
                    })
            return detections
        except Exception as exc:
            logger.error(f"Object detection error: {exc}")
            return []

    @property
    def last_inference_ms(self) -> float:
        return self._last_inference_ms


object_detector = ObjectDetector()
