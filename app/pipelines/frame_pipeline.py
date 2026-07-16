import logging
import uuid
import time
from datetime import datetime, timezone
from typing import Optional
import numpy as np
from app.ai.object_detector import object_detector
from app.ai.face_detector import face_detector
from app.ai.face_recognizer import face_recognizer
from app.ai.expression_estimator import expression_estimator
from app.ai.hand_landmarker import hand_landmarker
from app.ai.gesture_recognizer import gesture_recognizer
from app.ai.pose_estimator import pose_estimator
from app.ai.tracker import tracker
from app.ai.scene_understanding import scene_understanding
from app.ai.scene_captioner import scene_captioner
from app.pipelines.object_pipeline import run_object_pipeline
from app.pipelines.face_pipeline import run_face_pipeline
from app.pipelines.expression_pipeline import run_expression_pipeline
from app.pipelines.gesture_pipeline import run_gesture_pipeline
from app.pipelines.movement_pipeline import run_movement_pipeline
from app.pipelines.scene_pipeline import run_scene_pipeline

logger = logging.getLogger(__name__)


async def run_frame_pipeline(
    frame: np.ndarray,
    robot_id: str,
    features: dict,
    face_profiles: list[dict],
    gesture_profiles: list[dict],
) -> dict:
    """Run the full AI pipeline on one frame.

    Returns a structured vision state dict ready for Redis, WebSocket and Kafka.
    """
    t_start = time.monotonic()
    frame_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()

    detections: list[dict] = []
    face_results: list[dict] = []
    expression_results: list[dict] = []
    gesture_results: list[dict] = []
    movement_results: list[dict] = []
    scene: dict = {}
    description: str = ""
    low_light: bool = False

    # --- Object detection + tracking ---
    if features.get("object_detection"):
        detections = await run_object_pipeline(frame, features)

    # --- Face pipeline ---
    if features.get("face_detection"):
        face_results = await run_face_pipeline(
            frame, features, face_profiles, detections
        )

    # --- Expression pipeline ---
    if features.get("expression_estimation") and face_results:
        expression_results = await run_expression_pipeline(frame, face_results)

    # --- Gesture pipeline ---
    if features.get("gesture_detection"):
        gesture_results = await run_gesture_pipeline(frame, gesture_profiles)

    # --- Movement pipeline ---
    if features.get("person_tracking") and detections:
        movement_results = await run_movement_pipeline(detections)

    # --- Low-light check ---
    if features.get("low_light_detection"):
        gray_mean = float(np.mean(frame)) if frame is not None else 128.0
        low_light = gray_mean < 50.0

    # --- Scene pipeline ---
    if features.get("scene_understanding"):
        scene, description = await run_scene_pipeline(
            detections, face_results, gesture_results, movement_results, low_light, frame
        )

    pipeline_ms = (time.monotonic() - t_start) * 1000

    return {
        "frame_id": frame_id,
        "timestamp": timestamp,
        "robot_id": robot_id,
        "detections": detections,
        "faces": face_results,
        "expressions": expression_results,
        "gestures": gesture_results,
        "movements": movement_results,
        "scene": scene,
        "description": description,
        "low_light": low_light,
        "performance": {
            "pipeline_ms": round(pipeline_ms, 2),
            "fps": 0.0,  # Updated by caller over rolling window
            "gpu_memory_mb": _get_gpu_memory(),
        },
    }


def _get_gpu_memory() -> Optional[float]:
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            return round(torch.cuda.memory_allocated() / 1e6, 1)
    except Exception:
        pass
    return None
