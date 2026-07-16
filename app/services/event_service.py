import json
import logging
from datetime import datetime, timezone
from app.config.redis import get_redis
from app.config.settings import settings
from app.services.kafka_service import kafka_service

logger = logging.getLogger(__name__)


class EventService:
    """Manages vision event deduplication, cooldown, and Kafka publishing."""

    async def publish_if_not_cooldown(
        self,
        robot_id: str,
        event_type: str,
        topic: str,
        payload: dict,
        cooldown_seconds: int | None = None,
    ) -> bool:
        """Publish an event only if the cooldown window has expired."""
        cd = cooldown_seconds if cooldown_seconds is not None else settings.EVENT_COOLDOWN_SECONDS
        redis = await get_redis()
        cooldown_key = f"rex:vision:event-cooldown:{robot_id}:{event_type}"

        if await redis.get(cooldown_key):
            return False  # Still in cooldown

        full_payload = {
            "event_id": self._new_id(),
            "robot_id": robot_id,
            "event_type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **payload,
        }
        await kafka_service.publish(topic, full_payload)
        await redis.set(cooldown_key, "1", ex=cd)
        return True

    async def publish_always(self, topic: str, payload: dict) -> None:
        """Publish without cooldown (for state-update topics)."""
        await kafka_service.publish(topic, payload)

    @staticmethod
    def _new_id() -> str:
        import uuid
        return str(uuid.uuid4())


event_service = EventService()
