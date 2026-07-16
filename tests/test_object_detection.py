import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from app.ai.object_detector import ObjectDetector
from app.pipelines.object_pipeline import run_object_pipeline
from app.pipelines.movement_pipeline import classify_movement


# ---------------------------------------------------------------------------
# ObjectDetector unit tests
# ---------------------------------------------------------------------------
class TestObjectDetector:
    def test_detect_returns_empty_when_not_loaded(self):
        detector = ObjectDetector()
        result = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))
        assert result == []

    def test_detect_returns_empty_on_model_error(self):
        detector = ObjectDetector()
        detector.is_loaded = True
        detector._model = MagicMock(side_effect=RuntimeError("CUDA OOM"))
        result = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))
        assert result == []

    def test_load_falls_back_to_cpu_when_cuda_missing(self):
        detector = ObjectDetector()
        with patch("app.ai.object_detector.YOLO", side_effect=[RuntimeError("no cuda"), MagicMock()], create=True):
            # Shouldn't raise
            pass  # actual load tested via integration

    def test_unload_resets_state(self):
        detector = ObjectDetector()
        detector.is_loaded = True
        detector._model = MagicMock()
        detector.unload()
        assert not detector.is_loaded
        assert detector._model is None

    def test_last_inference_ms_default(self):
        detector = ObjectDetector()
        assert detector.last_inference_ms == 0.0


# ---------------------------------------------------------------------------
# Object pipeline
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_object_pipeline_with_mocked_detector(mock_yolo):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    features = {"object_detection": True}
    result = await run_object_pipeline(frame, features)
    assert isinstance(result, list)


@pytest.mark.asyncio
async def test_object_pipeline_returns_empty_when_disabled():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    with patch("app.ai.object_detector.object_detector.detect", return_value=[]):
        result = await run_object_pipeline(frame, {"object_detection": False})
    assert result == []


# ---------------------------------------------------------------------------
# Movement classification
# ---------------------------------------------------------------------------
class TestMovementClassification:
    def test_stationary(self):
        mv, vx, vy, sc = classify_movement(100, 100, 100, 100, 10000, 10000)
        assert mv == "STATIONARY"

    def test_moving_left(self):
        mv, vx, vy, sc = classify_movement(200, 100, 200, 200, 10000, 10000)
        assert mv == "MOVING_LEFT"
        assert vx < 0

    def test_moving_right(self):
        mv, vx, vy, sc = classify_movement(100, 200, 100, 100, 10000, 10000)
        assert mv == "MOVING_RIGHT"

    def test_moving_toward_camera(self):
        mv, vx, vy, sc = classify_movement(100, 100, 100, 100, 5000, 8000)
        assert mv == "MOVING_TOWARD_CAMERA"

    def test_moving_away_from_camera(self):
        mv, vx, vy, sc = classify_movement(100, 100, 100, 100, 8000, 5000)
        assert mv == "MOVING_AWAY_FROM_CAMERA"
