import logging
import numpy as np
from app.config.settings import settings

logger = logging.getLogger(__name__)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


class FaceRecognizer:
    """Generates 512-d face embeddings and classifies recognition results.
    
    Face detection + embedding generation is handled by FaceDetector (InsightFace).
    This class evaluates similarity against retrieved Qdrant embeddings.
    """

    def evaluate(
        self,
        query_embedding: list[float],
        candidate_embedding: list[float],
    ) -> dict:
        """Returns KNOWN / UNCERTAIN / UNKNOWN based on configured thresholds."""
        similarity = cosine_similarity(query_embedding, candidate_embedding)
        if similarity >= settings.FACE_KNOWN_THRESHOLD:
            status = "KNOWN"
        elif similarity >= settings.FACE_UNCERTAIN_THRESHOLD:
            status = "UNCERTAIN"
        else:
            status = "UNKNOWN"
        return {"similarity": round(similarity, 4), "status": status}

    def normalize_embedding(self, embedding: list[float]) -> list[float]:
        arr = np.array(embedding, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm == 0:
            return embedding
        return (arr / norm).tolist()


face_recognizer = FaceRecognizer()
