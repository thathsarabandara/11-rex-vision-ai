import json
import logging
from typing import Optional
from app.config.redis import get_redis

logger = logging.getLogger(__name__)


class TrackingService:
    async def set_target(self, robot_id: str, track_id: int) -> None:
        redis = await get_redis()
        await redis.set(
            f"rex:vision:target:{robot_id}",
            json.dumps({"track_id": track_id, "active": True}),
            ex=300,
        )
        logger.info(f"Tracking target set: robot={robot_id} track_id={track_id}")

    async def clear_target(self, robot_id: str) -> None:
        redis = await get_redis()
        await redis.delete(f"rex:vision:target:{robot_id}")
        logger.info(f"Tracking target cleared: robot={robot_id}")

    async def get_target(self, robot_id: str) -> Optional[dict]:
        redis = await get_redis()
        raw = await redis.get(f"rex:vision:target:{robot_id}")
        return json.loads(raw) if raw else None

    async def update_tracks(self, robot_id: str, tracks: list[dict]) -> None:
        redis = await get_redis()
        await redis.set(
            f"rex:vision:tracks:{robot_id}",
            json.dumps(tracks),
            ex=10,
        )


tracking_service = TrackingService()
