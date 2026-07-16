import logging
from app.ai.scene_captioner import scene_captioner
from app.ai.scene_understanding import scene_understanding

logger = logging.getLogger(__name__)


class SceneService:
    def build_scene(
        self,
        detections: list[dict],
        face_results: list[dict],
        gesture_results: list[dict],
        movement_results: list[dict],
        low_light: bool = False,
    ) -> dict:
        return scene_understanding.build(
            detections=detections,
            face_results=face_results,
            gesture_results=gesture_results,
            movement_results=movement_results,
            low_light=low_light,
        )

    def describe_scene(self, scene: dict, frame=None) -> str:
        return scene_captioner.describe(scene, frame=frame)


scene_service = SceneService()
