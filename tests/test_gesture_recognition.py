import pytest
import numpy as np
from app.ai.gesture_recognizer import GestureRecognizer
from app.utils.landmarks import normalize_hand_landmarks, landmarks_to_feature_vector
from app.utils.embeddings import cosine_similarity, normalize_vector, average_embeddings


class TestGestureRecognizer:
    def setup_method(self):
        self.recognizer = GestureRecognizer()

    def test_evaluate_similarity_identical_vectors(self):
        v = [0.1] * 63
        score = self.recognizer.evaluate_similarity(v, v)
        assert abs(score - 1.0) < 1e-4

    def test_evaluate_similarity_zero_vector(self):
        v1 = [0.0] * 63
        v2 = [1.0] * 63
        score = self.recognizer.evaluate_similarity(v1, v2)
        assert score == 0.0

    def test_stability_below_threshold_resets_buffer(self):
        source_id = "test-source"
        # Confidence below threshold → buffer reset
        stability = self.recognizer.update_stability(source_id, "THUMBS_UP", 0.01)
        assert stability == 0

    def test_stability_increases_with_consistent_gesture(self):
        source_id = "src-stable"
        for _ in range(6):
            self.recognizer.update_stability(source_id, "OPEN_PALM", 0.95)
        buf = self.recognizer._stability_buffer.get(source_id, [])
        assert buf.count("OPEN_PALM") >= 5

    def test_is_stable_returns_true_after_sufficient_frames(self):
        from app.config.settings import settings
        source_id = "src-stable-check"
        for _ in range(settings.GESTURE_STABILITY_FRAMES + 1):
            self.recognizer.update_stability(source_id, "STOP_GESTURE", 0.90)
        assert self.recognizer.is_stable(source_id, "STOP_GESTURE")

    def test_clear_source_removes_buffer(self):
        self.recognizer._stability_buffer["to-clear"] = ["THUMBS_UP"] * 5
        self.recognizer.clear_source("to-clear")
        assert "to-clear" not in self.recognizer._stability_buffer


class TestLandmarkUtils:
    def test_normalize_hand_landmarks_output_dim(self):
        coords = list(np.random.randn(63))
        result = normalize_hand_landmarks(coords)
        assert len(result) == 63

    def test_normalize_centres_on_wrist(self):
        """After normalization, the wrist (first landmark) should be at origin."""
        coords = [0.5] * 63  # all same value
        result = normalize_hand_landmarks(coords)
        # Wrist components (first 3) should be zero
        assert abs(result[0]) < 1e-5
        assert abs(result[1]) < 1e-5
        assert abs(result[2]) < 1e-5

    def test_landmarks_to_feature_vector_output_dim(self):
        lm_list = [{"x": i * 0.01, "y": i * 0.02, "z": 0.0} for i in range(21)]
        v = landmarks_to_feature_vector(lm_list)
        assert len(v) == 63


class TestEmbeddingUtils:
    def test_cosine_similarity_same_vector(self):
        v = list(np.random.randn(512))
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-5

    def test_normalize_vector_unit_length(self):
        v = [3.0, 4.0]
        nv = normalize_vector(v)
        assert abs(sum(x**2 for x in nv) ** 0.5 - 1.0) < 1e-5

    def test_average_embeddings_returns_empty_on_empty_input(self):
        assert average_embeddings([]) == []

    def test_average_embeddings_returns_normalized_result(self):
        v = [[1.0, 0.0], [1.0, 0.0]]
        result = average_embeddings(v)
        assert len(result) == 2
        assert abs(sum(x**2 for x in result) ** 0.5 - 1.0) < 1e-5
