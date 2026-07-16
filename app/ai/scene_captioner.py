import logging

logger = logging.getLogger(__name__)


class SceneCaptioner:
    """Produces a human-readable scene description.

    Two modes:
    - RULE_BASED (always available): generates a sentence from structured scene state.
    - LIGHTWEIGHT_CAPTION_MODEL (optional, disabled by default): uses a small
      image-captioning model when loaded.

    The rule-based mode is the default and is always preferred in production
    unless a lightweight model is explicitly configured.
    """

    def __init__(self) -> None:
        self._caption_model = None
        self.is_loaded: bool = False
        self.mode: str = "RULE_BASED"

    def load_caption_model(self, model_path: str) -> bool:
        """Optionally load a lightweight captioning model."""
        try:
            # Placeholder: integrate a small BLIP or GIT model here.
            # from transformers import BlipProcessor, BlipForConditionalGeneration
            # self._caption_model = ...
            logger.info(f"Caption model path configured: {model_path}")
            self.is_loaded = True
            self.mode = "LIGHTWEIGHT_CAPTION_MODEL"
            return True
        except Exception as exc:
            logger.warning(f"Caption model failed to load: {exc}. Falling back to RULE_BASED.")
            self.mode = "RULE_BASED"
            return False

    def describe(self, scene: dict, frame=None) -> str:
        """Generate a sentence describing the current scene."""
        if self.mode == "LIGHTWEIGHT_CAPTION_MODEL" and self._caption_model and frame is not None:
            try:
                return self._describe_with_model(frame)
            except Exception as exc:
                logger.warning(f"Caption model inference failed: {exc}. Using rule-based.")

        return self._describe_rule_based(scene)

    def _describe_rule_based(self, scene: dict) -> str:
        parts: list[str] = []

        person_count = scene.get("person_count", 0)
        known_faces = scene.get("known_faces", [])
        unknown_count = scene.get("unknown_face_count", 0)
        objects = scene.get("important_objects", [])
        activities = scene.get("activities", [])
        low_light = scene.get("low_light", False)
        security = scene.get("security_observation", "")
        pets = scene.get("pet_count", 0)

        if person_count == 0:
            parts.append("No people are visible.")
        elif person_count == 1:
            name = known_faces[0] if known_faces else "one unknown person"
            parts.append(f"One person is visible ({name}).")
        else:
            parts.append(f"{person_count} people are visible.")
            if known_faces:
                parts.append(f"Recognised: {', '.join(known_faces)}.")
            if unknown_count > 0:
                parts.append(f"{unknown_count} unknown {'person' if unknown_count == 1 else 'people'} present.")

        if pets > 0:
            parts.append(f"{pets} pet{'s' if pets > 1 else ''} visible.")

        if objects:
            parts.append(f"Nearby objects: {', '.join(objects[:5])}.")

        if activities:
            parts.append(" ".join(activities[:3]).capitalize() + ".")

        if low_light:
            parts.append("Low-light conditions detected.")

        if security:
            parts.append(f"Security observation: {security}.")

        return " ".join(parts) if parts else "Scene is empty."

    def _describe_with_model(self, frame) -> str:
        # Placeholder for actual model inference
        return "Scene description from model not yet implemented."


scene_captioner = SceneCaptioner()
