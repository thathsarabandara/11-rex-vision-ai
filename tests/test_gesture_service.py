import pytest
from unittest.mock import AsyncMock, patch
from app.services.gesture_service import GestureService


class TestGestureService:
    def setup_method(self):
        self.service = GestureService()

    @pytest.mark.asyncio
    async def test_register_gesture_rejects_bad_mime(self, mock_qdrant, mock_minio):
        from tests.conftest import TestSessionLocal
        async with TestSessionLocal() as db:
            with pytest.raises(ValueError, match="Unsupported image type"):
                await self.service.register_gesture(
                    robot_id="robot-001", user_id="user-001",
                    gesture_name="THUMBS_UP", description=None,
                    gesture_type="STATIC", action_hint=None,
                    sample_images=[b"fake"], content_types=["application/pdf"],
                    db=db,
                )

    @pytest.mark.asyncio
    async def test_register_gesture_rejects_no_hands(self, mock_qdrant, mock_minio):
        from tests.conftest import TestSessionLocal
        with patch("app.services.gesture_service.GestureService._extract_vectors", return_value=[]):
            async with TestSessionLocal() as db:
                with pytest.raises(ValueError, match="No valid hand landmarks"):
                    await self.service.register_gesture(
                        robot_id="robot-001", user_id="user-001",
                        gesture_name="TEST", description=None,
                        gesture_type="STATIC", action_hint=None,
                        sample_images=[b"fake"], content_types=["image/jpeg"],
                        db=db,
                    )

    @pytest.mark.asyncio
    async def test_list_profiles_empty(self, mock_qdrant):
        from tests.conftest import TestSessionLocal
        async with TestSessionLocal() as db:
            profiles = await self.service.list_profiles("robot-no-gestures", db)
            assert profiles == []

    @pytest.mark.asyncio
    async def test_get_nonexistent_profile_returns_none(self, mock_qdrant):
        from tests.conftest import TestSessionLocal
        async with TestSessionLocal() as db:
            profile = await self.service.get_profile("robot-001", "nonexistent", db)
            assert profile is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_profile_returns_false(self, mock_qdrant, mock_minio):
        from tests.conftest import TestSessionLocal
        async with TestSessionLocal() as db:
            result = await self.service.delete_profile("robot-001", "nonexistent", db)
            assert result is False

    @pytest.mark.asyncio
    async def test_register_gesture_full_flow(self, mock_qdrant, mock_minio):
        from tests.conftest import TestSessionLocal
        import numpy as np
        fake_vector = list(np.zeros(63))
        with patch("app.services.gesture_service.GestureService._extract_vectors", return_value=[fake_vector]):
            async with TestSessionLocal() as db:
                profile = await self.service.register_gesture(
                    robot_id="robot-g-001", user_id="user-001",
                    gesture_name="OPEN_PALM", description="Open hand",
                    gesture_type="STATIC", action_hint="pause",
                    sample_images=[b"fake"], content_types=["image/jpeg"],
                    db=db,
                )
                assert profile.gesture_name == "OPEN_PALM"
                assert profile.action_hint == "pause"
                assert profile.sample_count == 1

    @pytest.mark.asyncio
    async def test_update_gesture_profile(self, mock_qdrant, mock_minio):
        from tests.conftest import TestSessionLocal
        import numpy as np
        with patch("app.services.gesture_service.GestureService._extract_vectors", return_value=[list(np.zeros(63))]):
            async with TestSessionLocal() as db:
                profile = await self.service.register_gesture(
                    robot_id="robot-g-002", user_id="user-001",
                    gesture_name="FIST", description=None,
                    gesture_type="STATIC", action_hint=None,
                    sample_images=[b"fake"], content_types=["image/jpeg"],
                    db=db,
                )
                updated = await self.service.update_profile(
                    "robot-g-002", profile.gesture_profile_id,
                    {"gesture_name": "CLOSED_FIST", "description": "Closed fist"}, db,
                )
                assert updated.gesture_name == "CLOSED_FIST"
