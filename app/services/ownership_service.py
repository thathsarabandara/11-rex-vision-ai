import httpx
import logging
from app.config.settings import settings
from app.config.redis import get_redis

logger = logging.getLogger(__name__)


class OwnershipService:
    async def verify_ownership(self, user_id: str, robot_id: str) -> bool:
        redis = await get_redis()
        cache_key = f"rex:vision:ownership:{user_id}:{robot_id}"
        cached = await redis.get(cache_key)
        if cached is not None:
            return cached == "1"

        try:
            async with httpx.AsyncClient() as client:
                headers = {"Authorization": f"Bearer {settings.INTERNAL_SERVICE_TOKEN}"}
                url = f"{settings.ROBOT_SERVICE_URL}/internal/v1/robots/{robot_id}/ownership/{user_id}"
                response = await client.get(url, headers=headers, timeout=5.0)
                if response.status_code == 200:
                    owned = response.json().get("owned", False)
                    await redis.set(cache_key, "1" if owned else "0", ex=settings.OWNERSHIP_CACHE_TTL_SECONDS)
                    return owned
                logger.warning(f"Ownership check returned {response.status_code}")
                return False
        except Exception as exc:
            logger.error(f"Ownership check error: {exc}")
            return False


ownership_service = OwnershipService()
