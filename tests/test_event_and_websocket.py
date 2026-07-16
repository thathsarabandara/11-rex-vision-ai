import pytest
from app.services.event_service import EventService
from app.services.websocket_service import WebSocketService


class TestEventService:
    def setup_method(self):
        self.service = EventService()

    @pytest.mark.asyncio
    async def test_publish_if_not_cooldown_fires_first_time(self):
        from unittest.mock import AsyncMock, patch
        with patch("app.services.event_service.kafka_service.publish", new=AsyncMock()) as mock_pub:
            fired = await self.service.publish_if_not_cooldown(
                "robot-evt-001", "test.event", "rex.vision.test",
                {"detail": "ok"}, cooldown_seconds=5,
            )
            assert fired is True
            mock_pub.assert_called_once()

    @pytest.mark.asyncio
    async def test_publish_respects_cooldown(self):
        from unittest.mock import AsyncMock, patch
        from app.config.redis import get_redis
        robot_id = "robot-cooldown-001"
        event_type = "cooldown.test"
        redis = await get_redis()
        await redis.set(f"rex:vision:event-cooldown:{robot_id}:{event_type}", "1", ex=60)

        with patch("app.services.event_service.kafka_service.publish", new=AsyncMock()) as mock_pub:
            fired = await self.service.publish_if_not_cooldown(
                robot_id, event_type, "rex.vision.test",
                {}, cooldown_seconds=60,
            )
            assert fired is False
            mock_pub.assert_not_called()

    @pytest.mark.asyncio
    async def test_publish_always_always_fires(self):
        from unittest.mock import AsyncMock, patch
        with patch("app.services.event_service.kafka_service.publish", new=AsyncMock()) as mock_pub:
            await self.service.publish_always("rex.vision.state", {"data": "test"})
            mock_pub.assert_called_once()


class TestWebSocketService:
    def setup_method(self):
        self.service = WebSocketService()

    def test_connection_count_zero_initially(self):
        assert self.service.connection_count("no-robot") == 0

    def test_authenticate_invalid_token_returns_none(self):
        result = self.service.authenticate("invalid.jwt.token")
        assert result is None

    def test_authenticate_valid_token(self):
        import jwt
        from app.config.settings import settings
        payload = {
            "sub": "u-001", "email_verified": True,
            "iss": settings.USER_JWT_ISSUER,
            "aud": settings.USER_JWT_AUDIENCE,
        }
        token = jwt.encode(payload, settings.USER_JWT_SECRET_KEY, algorithm=settings.USER_JWT_ALGORITHM)
        result = self.service.authenticate(token)
        assert result is not None
        assert result["sub"] == "u-001"

    def test_authenticate_unverified_email_returns_none(self):
        import jwt
        from app.config.settings import settings
        payload = {
            "sub": "u-002", "email_verified": False,
            "iss": settings.USER_JWT_ISSUER,
            "aud": settings.USER_JWT_AUDIENCE,
        }
        token = jwt.encode(payload, settings.USER_JWT_SECRET_KEY, algorithm=settings.USER_JWT_ALGORITHM)
        result = self.service.authenticate(token)
        assert result is None
