import pytest
from unittest.mock import AsyncMock, patch
from app.services.vision_service import VisionService, validate_feature_dependencies, DEFAULT_FEATURES


class TestFeatureDependencyValidation:
    def test_no_error_for_valid_config(self):
        features = {
            "object_detection": True, "face_detection": True,
            "face_recognition": True, "expression_estimation": True,
            "person_tracking": True, "gesture_detection": True,
            "scene_understanding": True, "scene_description": True,
            "visual_obstacle_detection": False, "low_light_detection": True,
            "incident_capture": False,
        }
        assert validate_feature_dependencies(features) is None

    def test_face_recognition_requires_face_detection(self):
        features = {"face_recognition": True, "face_detection": False}
        error = validate_feature_dependencies(features)
        assert error is not None
        assert "face_detection" in error

    def test_expression_estimation_requires_face_detection(self):
        features = {"expression_estimation": True, "face_detection": False}
        error = validate_feature_dependencies(features)
        assert error is not None
        assert "face_detection" in error

    def test_person_tracking_requires_object_detection(self):
        features = {"person_tracking": True, "object_detection": False}
        error = validate_feature_dependencies(features)
        assert error is not None
        assert "object_detection" in error

    def test_scene_description_requires_scene_understanding(self):
        features = {"scene_description": True, "scene_understanding": False}
        error = validate_feature_dependencies(features)
        assert error is not None

    def test_all_disabled_no_error(self):
        features = {k: False for k in DEFAULT_FEATURES}
        assert validate_feature_dependencies(features) is None


class TestVisionServiceFeatures:
    def setup_method(self):
        self.service = VisionService()

    @pytest.mark.asyncio
    async def test_get_features_from_db_when_cache_miss(self):
        from tests.conftest import TestSessionLocal
        async with TestSessionLocal() as db:
            features = await self.service.get_features("robot-vs-001", db)
            assert "object_detection" in features
            assert "face_detection" in features

    @pytest.mark.asyncio
    async def test_update_features_validates_dependencies(self):
        from tests.conftest import TestSessionLocal
        async with TestSessionLocal() as db:
            with pytest.raises(ValueError):
                await self.service.update_features(
                    "robot-vs-001",
                    {"face_recognition": True, "face_detection": False},
                    db,
                )

    @pytest.mark.asyncio
    async def test_get_latest_state_returns_none_for_unknown_robot(self):
        result = await self.service.get_latest_state("no-such-robot")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_or_create_config_idempotent(self):
        from tests.conftest import TestSessionLocal
        async with TestSessionLocal() as db:
            config1 = await self.service.get_or_create_config("robot-idem-001", db)
            config2 = await self.service.get_or_create_config("robot-idem-001", db)
            assert config1.robot_id == config2.robot_id

    @pytest.mark.asyncio
    async def test_create_camera_source(self):
        from tests.conftest import TestSessionLocal
        async with TestSessionLocal() as db:
            source = await self.service.create_camera_source("robot-cam-001", {
                "source_type": "USB_CAMERA",
                "source_url": None,
                "enabled": True,
                "resolution": "1280x720",
                "target_fps": 30,
            }, db)
            assert source.source_type == "USB_CAMERA"
            assert source.target_fps == 30
