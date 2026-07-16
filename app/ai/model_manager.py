import logging
from datetime import datetime
from typing import Optional
from app.config.settings import settings
from app.ai.object_detector import object_detector
from app.ai.face_detector import face_detector
from app.ai.expression_estimator import expression_estimator
from app.ai.hand_landmarker import hand_landmarker
from app.ai.gesture_recognizer import gesture_recognizer
from app.ai.pose_estimator import pose_estimator
from app.ai.tracker import tracker
from app.ai.scene_captioner import scene_captioner

logger = logging.getLogger(__name__)

MODEL_KEYS = [
    "object-detector",
    "face-detector",
    "face-recognizer",
    "expression-estimator",
    "gesture-recognizer",
    "pose-estimator",
    "scene-captioner",
]


class ModelManager:
    """Central registry for all AI model instances.

    Manages lazy loading, unloading, and runtime status reporting.
    Prevents disabling a model that is a dependency of an enabled feature.
    """

    DEPENDENCIES: dict[str, list[str]] = {
        "face-recognizer": ["face-detector"],
        "expression-estimator": ["face-detector"],
        "gesture-recognizer": ["gesture-recognizer"],  # self-sufficient
    }

    def __init__(self) -> None:
        self._status: dict[str, dict] = {
            key: {
                "enabled": True,
                "loaded": False,
                "device": settings.DEVICE,
                "status": "DISABLED",
                "last_loaded_at": None,
                "average_inference_ms": None,
            }
            for key in MODEL_KEYS
        }

    def _get_wrapper(self, model_key: str):
        mapping = {
            "object-detector": object_detector,
            "face-detector": face_detector,
            "face-recognizer": None,  # embedded in face_detector
            "expression-estimator": expression_estimator,
            "gesture-recognizer": None,  # logic-only, no load needed
            "pose-estimator": pose_estimator,
            "scene-captioner": scene_captioner,
        }
        return mapping.get(model_key)

    def load_model(self, model_key: str) -> bool:
        wrapper = self._get_wrapper(model_key)
        if wrapper is None:
            # Logic-only model, mark as loaded
            self._status[model_key]["loaded"] = True
            self._status[model_key]["status"] = "READY"
            self._status[model_key]["last_loaded_at"] = datetime.utcnow().isoformat()
            return True

        self._status[model_key]["status"] = "LOADING"
        success = wrapper.load() if hasattr(wrapper, "load") else True
        if success:
            self._status[model_key]["loaded"] = True
            self._status[model_key]["status"] = "READY"
            self._status[model_key]["last_loaded_at"] = datetime.utcnow().isoformat()
        else:
            self._status[model_key]["status"] = "FAILED"
        return success

    def unload_model(self, model_key: str) -> None:
        wrapper = self._get_wrapper(model_key)
        if wrapper and hasattr(wrapper, "unload"):
            wrapper.unload()
        self._status[model_key]["loaded"] = False
        self._status[model_key]["status"] = "DISABLED"

    def set_enabled(self, model_key: str, enabled: bool) -> tuple[bool, str]:
        if model_key not in self._status:
            return False, f"Unknown model key: {model_key}"

        if not enabled:
            # Check that no enabled dependent models rely on this one
            for dep_key, deps in self.DEPENDENCIES.items():
                if model_key in deps and self._status.get(dep_key, {}).get("enabled"):
                    return False, f"Cannot disable {model_key}: {dep_key} depends on it"
            self.unload_model(model_key)
            self._status[model_key]["enabled"] = False
        else:
            self._status[model_key]["enabled"] = True
            self.load_model(model_key)

        return True, "OK"

    def get_status(self, model_key: Optional[str] = None) -> list[dict]:
        if model_key:
            info = self._status.get(model_key)
            return [{"model_key": model_key, **info}] if info else []
        return [{"model_key": k, **v} for k, v in self._status.items()]

    def is_ready(self, model_key: str) -> bool:
        return self._status.get(model_key, {}).get("status") == "READY"

    def load_all_enabled(self) -> None:
        """Load all enabled models at startup."""
        for key in MODEL_KEYS:
            if self._status[key]["enabled"]:
                self.load_model(key)


model_manager = ModelManager()
