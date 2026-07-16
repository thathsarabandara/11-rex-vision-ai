import logging
import numpy as np
from app.ai.scene_understanding import scene_understanding
from app.ai.scene_captioner import scene_captioner

logger = logging.getLogger(__name__)


async def run_scene_pipeline(
    detections: list[dict],
    face_results: list[dict],
    gesture_results: list[dict],
    movement_results: list[dict],
    low_light: bool,
    frame: "np.ndarray | None" = None,
) -> tuple[dict, str]:
    """Build structured scene state and generate a description."""
    scene = scene_understanding.build(
        detections=detections,
        face_results=face_results,
        gesture_results=gesture_results,
        movement_results=movement_results,
        low_light=low_light,
    )
    description = scene_captioner.describe(scene, frame=frame)
    return scene, description
