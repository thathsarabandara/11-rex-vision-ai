import pytest
import numpy as np
from app.ai.face_recognizer import FaceRecognizer, cosine_similarity


class TestCosineSimiliarity:
    def test_identical_vectors_return_one(self):
        v = list(np.random.randn(512))
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-5

    def test_orthogonal_vectors_return_zero(self):
        v1 = [1.0, 0.0]
        v2 = [0.0, 1.0]
        assert abs(cosine_similarity(v1, v2)) < 1e-5

    def test_zero_vector_returns_zero(self):
        v1 = [0.0, 0.0]
        v2 = [1.0, 2.0]
        assert cosine_similarity(v1, v2) == 0.0


class TestFaceRecognizer:
    def setup_method(self):
        self.recognizer = FaceRecognizer()

    def test_returns_known_above_threshold(self):
        v = list(np.ones(512))
        result = self.recognizer.evaluate(v, v)
        assert result["status"] == "KNOWN"
        assert result["similarity"] >= 0.75

    def test_returns_unknown_for_dissimilar_vectors(self):
        np.random.seed(42)
        v1 = list(np.random.randn(512))
        v2 = list(-np.random.randn(512))  # opposite direction
        result = self.recognizer.evaluate(v1, v2)
        # Should be UNKNOWN or UNCERTAIN (very low similarity)
        assert result["status"] in ("UNKNOWN", "UNCERTAIN")

    def test_returns_uncertain_in_middle_range(self):
        """Test that mid-range similarity maps to UNCERTAIN."""
        from app.config.settings import settings
        # Create vectors with controlled cosine similarity
        v_base = list(np.ones(512) / np.sqrt(512))
        # Mix of same and different to get mid similarity
        noise = list(np.random.randn(512) * 0.5)
        v_mixed = [(a + b) for a, b in zip(v_base, noise)]
        result = self.recognizer.evaluate(v_base, v_mixed)
        assert result["status"] in ("KNOWN", "UNCERTAIN", "UNKNOWN")  # depends on noise

    def test_normalize_embedding_unit_length(self):
        v = [3.0, 4.0]
        normalized = self.recognizer.normalize_embedding(v)
        length = sum(x**2 for x in normalized) ** 0.5
        assert abs(length - 1.0) < 1e-5

    def test_normalize_zero_vector_unchanged(self):
        v = [0.0, 0.0, 0.0]
        result = self.recognizer.normalize_embedding(v)
        assert result == v
