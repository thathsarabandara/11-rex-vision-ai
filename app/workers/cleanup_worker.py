import asyncio
import logging
from datetime import datetime, timezone, timedelta
from sqlalchemy.future import select
from app.config.database import AsyncSessionLocal
from app.models.vision_event import VisionEvent
from app.models.incident_media import IncidentMedia

logger = logging.getLogger(__name__)

CLEANUP_INTERVAL_SECONDS = 3600  # Run every hour
EVENT_RETENTION_DAYS = 90
INCIDENT_RETENTION_DAYS = 30

_cleanup_task: asyncio.Task | None = None


async def _cleanup_loop() -> None:
    while True:
        try:
            await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
            await _run_cleanup()
        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.error(f"Cleanup worker error: {exc}")


async def _run_cleanup() -> None:
    logger.info("Running periodic cleanup")
    async with AsyncSessionLocal() as db:
        # Purge old vision events
        cutoff_events = datetime.now(timezone.utc) - timedelta(days=EVENT_RETENTION_DAYS)
        result = await db.execute(
            select(VisionEvent).where(VisionEvent.occurred_at < cutoff_events)
        )
        old_events = result.scalars().all()
        for evt in old_events:
            await db.delete(evt)

        # Purge old incident media records
        cutoff_media = datetime.now(timezone.utc) - timedelta(days=INCIDENT_RETENTION_DAYS)
        result2 = await db.execute(
            select(IncidentMedia).where(IncidentMedia.captured_at < cutoff_media)
        )
        old_media = result2.scalars().all()
        for media in old_media:
            # Best-effort MinIO deletion
            try:
                from app.services.storage_service import storage_service
                await storage_service.delete_object(media.minio_key)
            except Exception as exc:
                logger.warning(f"Failed to delete incident media from MinIO: {exc}")
            await db.delete(media)

        await db.commit()
    logger.info(f"Cleanup done: removed {len(old_events)} events, {len(old_media)} media records")


async def start_cleanup_worker() -> None:
    global _cleanup_task
    _cleanup_task = asyncio.create_task(_cleanup_loop())
    logger.info("Cleanup worker started")


async def stop_cleanup_worker() -> None:
    global _cleanup_task
    if _cleanup_task:
        _cleanup_task.cancel()
        try:
            await _cleanup_task
        except asyncio.CancelledError:
            pass
