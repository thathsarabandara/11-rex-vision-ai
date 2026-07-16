import pytest
from unittest.mock import AsyncMock, patch
from app.services.tracking_service import TrackingService


class TestTrackingService:
    def setup_method(self):
        self.service = TrackingService()

    @pytest.mark.asyncio
    async def test_set_and_get_target(self):
        await self.service.set_target("robot-track-001", 42)
        target = await self.service.get_target("robot-track-001")
        assert target is not None
        assert target["track_id"] == 42
        assert target["active"] is True

    @pytest.mark.asyncio
    async def test_clear_target(self):
        await self.service.set_target("robot-track-002", 10)
        await self.service.clear_target("robot-track-002")
        target = await self.service.get_target("robot-track-002")
        assert target is None

    @pytest.mark.asyncio
    async def test_get_target_returns_none_when_not_set(self):
        target = await self.service.get_target("robot-no-target")
        assert target is None
