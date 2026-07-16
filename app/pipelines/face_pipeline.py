import logging
import numpy as np
from app.ai.face_detector import face_detector
from app.ai.face_recognizer import face_recognizer
from app.config.settings import settings

logger = logging.getLogger(__name__)


async def run_face_pipeline(
    frame: np.ndarray,
    features: dict,
    face_profiles: list[dict],
    detections: list[dict],
) -> list[dict]:
    """Detect faces and optionally recognise them against registered profiles."""
    raw_faces = face_detector.detect(frame)
    results = []

    for i, face in enumerate(raw_faces):
        det_score = face.get("det_score", 0.0)
        embedding = face.get("embedding")
        bbox = face.get("bbox", {})

        # Reject low quality detections
        if det_score < 0.5:
            continue

        result: dict = {
            "face_index": i,
            "bbox": bbox,
            "det_score": round(det_score, 4),
            "status": "UNKNOWN",
            "face_profile_id": None,
            "display_name": None,
            "relationship": None,
            "similarity": None,
        }

        if features.get("face_recognition") and embedding and face_profiles:
            best_similarity = 0.0
            best_profile = None

            for profile in face_profiles:
                profile_embedding = profile.get("embedding")
                if not profile_embedding:
                    continue
                eval_result = face_recognizer.evaluate(embedding, profile_embedding)
                if eval_result["similarity"] > best_similarity:
                    best_similarity = eval_result["similarity"]
                    best_status = eval_result["status"]
                    best_profile = profile

            if best_profile and best_similarity >= settings.FACE_UNCERTAIN_THRESHOLD:
                result.update({
                    "status": best_status,
                    "face_profile_id": best_profile.get("face_profile_id"),
                    "display_name": best_profile.get("display_name"),
                    "relationship": best_profile.get("relationship"),
                    "similarity": round(best_similarity, 4),
                })

        results.append(result)

    return results
