import logging

logger = logging.getLogger(__name__)


class SceneUnderstanding:
    """Rule-based scene state aggregator.

    Combines object detections, face results, gesture results, and movement
    data into a structured scene dictionary. No neural network required.
    """

    def build(
        self,
        detections: list[dict],
        face_results: list[dict],
        gesture_results: list[dict],
        movement_results: list[dict],
        low_light: bool = False,
    ) -> dict:
        person_count = sum(1 for d in detections if d.get("class_name") == "person")
        pet_count = sum(1 for d in detections if d.get("class_name") in ("cat", "dog"))

        known_faces = [
            f["display_name"]
            for f in face_results
            if f.get("status") == "KNOWN" and f.get("display_name")
        ]
        unknown_face_count = sum(1 for f in face_results if f.get("status") == "UNKNOWN")

        important_objects = list({
            d["class_name"]
            for d in detections
            if d.get("class_name") not in ("person", "cat", "dog")
        })

        activities: list[str] = []
        for mv in movement_results:
            mv_type = mv.get("movement", "UNKNOWN")
            tid = mv.get("track_id")
            if mv_type not in ("STATIONARY", "UNKNOWN"):
                activities.append(f"track {tid} {mv_type.lower().replace('_', ' ')}")

        detected_gestures = [g.get("gesture_name", "") for g in gesture_results if g.get("gesture_name")]

        security_observation = ""
        if unknown_face_count > 0:
            security_observation = "unknown person present"

        return {
            "person_count": person_count,
            "pet_count": pet_count,
            "known_faces": known_faces,
            "unknown_face_count": unknown_face_count,
            "important_objects": important_objects,
            "activities": activities,
            "detected_gestures": detected_gestures,
            "low_light": low_light,
            "security_observation": security_observation,
        }


scene_understanding = SceneUnderstanding()
