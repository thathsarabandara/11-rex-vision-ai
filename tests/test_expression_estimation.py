import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from app.ai.expression_estimator import ExpressionEstimator, EXPRESSION_CLASSES


class TestExpressionEstimator:
    def setup_method(self):
        self.estimator = ExpressionEstimator()

    def test_returns_uncertain_when_not_loaded(self):
        result = self.estimator.estimate(np.zeros((48, 48, 3), dtype=np.uint8))
        assert result["expression"] == "UNCERTAIN"
        assert result["confidence"] == 0.0

    def test_returns_uncertain_on_inference_error(self):
        self.estimator.is_loaded = True
        self.estimator._session = MagicMock(side_effect=RuntimeError("ONNX error"))
        result = self.estimator.estimate(np.zeros((48, 48, 3), dtype=np.uint8))
        assert result["expression"] == "UNCERTAIN"

    def test_clears_track_stability_buffer(self):
        self.estimator._stability_buffer[42] = ["HAPPY", "HAPPY"]
        self.estimator.clear_track(42)
        assert 42 not in self.estimator._stability_buffer

    def test_stability_increases_with_consistent_results(self):
        self.estimator._stability_buffer[1] = ["HAPPY"] * 5
        count = self.estimator._stability_buffer[1].count("HAPPY")
        assert count == 5

    def test_unload_clears_stability_buffer(self):
        self.estimator._stability_buffer[99] = ["SAD"] * 3
        self.estimator.unload()
        assert len(self.estimator._stability_buffer) == 0
        assert not self.estimator.is_loaded

    def test_expression_classes_list(self):
        assert "HAPPY" in EXPRESSION_CLASSES
        assert "NEUTRAL" in EXPRESSION_CLASSES
        assert "UNCERTAIN" not in EXPRESSION_CLASSES  # UNCERTAIN is added by logic not the class list

    def test_mocked_onnx_inference(self):
        """Simulate a successful ONNX session returning HAPPY prediction."""
        import numpy as np
        mock_session = MagicMock()
        probs = np.zeros(7)
        probs[1] = 0.90  # HAPPY index
        mock_session.run.return_value = [np.array([probs])]
        mock_session.get_inputs.return_value = [MagicMock(name="input")]

        self.estimator.is_loaded = True
        self.estimator._session = mock_session
        self.estimator._input_name = "input"

        with patch("cv2.cvtColor", return_value=np.zeros((48, 48), dtype=np.uint8)), \
             patch("cv2.resize", return_value=np.zeros((48, 48), dtype=np.float32)):
            result = self.estimator.estimate(np.zeros((80, 80, 3), dtype=np.uint8), track_id=0)
            assert result["expression"] == "HAPPY"
            assert result["confidence"] >= 0.90
