import pytest
import numpy as np
from unittest.mock import patch
from app.pipelines.face_pipeline import run_face_pipeline
from app.pipelines.expression_pipeline import run_expression_pipeline
from app.pipelines.gesture_pipeline import run_gesture_pipeline
from app.pipelines.movement_pipeline import run_movement_pipeline


class TestFacePipeline:
    @pytest.mark.asyncio
    async def test_returns_empty_without_face_detection(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with patch("app.pipelines.face_pipeline.face_detector.detect", return_value=[]):
            result = await run_face_pipeline(frame, {"face_detection": True}, [], [])
        assert result == []

    @pytest.mark.asyncio
    async def test_rejects_low_confidence_faces(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        low_conf_face = {
            "bbox": {"x": 10, "y": 10, "width": 50, "height": 50},
            "det_score": 0.2,  # below 0.5 threshold
            "kps": None, "embedding": [0.0] * 512,
        }
        with patch("app.pipelines.face_pipeline.face_detector.detect", return_value=[low_conf_face]):
            result = await run_face_pipeline(frame, {"face_detection": True}, [], [])
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_unknown_status_without_profiles(self, mock_insightface):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = await run_face_pipeline(frame, {"face_detection": True, "face_recognition": True}, [], [])
        if result:
            assert result[0]["status"] == "UNKNOWN"

    @pytest.mark.asyncio
    async def test_face_recognition_matches_profile(self):
        import numpy as np
        from app.ai.face_recognizer import face_recognizer
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        emb = list(np.ones(512) / np.sqrt(512))
        face_data = {
            "bbox": {"x": 100, "y": 100, "width": 80, "height": 80},
            "det_score": 0.95, "kps": None, "embedding": emb,
        }
        profile = {
            "face_profile_id": "fp-001", "display_name": "Thathsara",
            "relationship": "OWNER", "embedding": emb,
        }
        with patch("app.pipelines.face_pipeline.face_detector.detect", return_value=[face_data]):
            result = await run_face_pipeline(
                frame, {"face_detection": True, "face_recognition": True},
                [profile], [],
            )
        assert len(result) == 1
        assert result[0]["status"] == "KNOWN"
        assert result[0]["display_name"] == "Thathsara"


class TestExpressionPipeline:
    @pytest.mark.asyncio
    async def test_skips_faces_with_zero_size_bbox(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        faces = [{"bbox": {"x": 0, "y": 0, "width": 0, "height": 0}}]
        result = await run_expression_pipeline(frame, faces)
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_uncertain_when_estimator_disabled(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        faces = [{"bbox": {"x": 100, "y": 100, "width": 80, "height": 80}}]
        with patch("app.pipelines.expression_pipeline.expression_estimator.estimate",
                   return_value={"expression": "UNCERTAIN", "confidence": 0.0, "stability_frames": 0}):
            result = await run_expression_pipeline(frame, faces)
        assert len(result) == 1
        assert result[0]["expression"] == "UNCERTAIN"


class TestGesturePipeline:
    @pytest.mark.asyncio
    async def test_returns_empty_without_hands(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with patch("app.pipelines.gesture_pipeline.hand_landmarker.extract", return_value=[]):
            result = await run_gesture_pipeline(frame, [])
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_without_profiles(self, mock_mediapipe):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = await run_gesture_pipeline(frame, [])
        assert result == []


class TestMovementPipeline:
    @pytest.mark.asyncio
    async def test_skips_non_person_detections(self):
        detections = [{"class_name": "dog", "track_id": 1}]
        result = await run_movement_pipeline(detections)
        assert result == []

    @pytest.mark.asyncio
    async def test_skips_person_without_track_id(self):
        detections = [{"class_name": "person", "track_id": None}]
        result = await run_movement_pipeline(detections)
        assert result == []

    @pytest.mark.asyncio
    async def test_skips_person_with_short_history(self):
        detections = [{"class_name": "person", "track_id": 5}]
        with patch("app.pipelines.movement_pipeline.tracker.get_track_history", return_value=[
            {"x": 100, "y": 100, "width": 50, "height": 100, "confidence": 0.9}
        ]):  # only 1 entry, need 3
            result = await run_movement_pipeline(detections)
        assert result == []
