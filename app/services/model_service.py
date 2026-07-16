import logging
from datetime import datetime, timezone
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.ai.model_manager import model_manager
from app.models.model_version import ModelVersion
from app.services.kafka_service import kafka_service
from app.config.kafka import TOPIC_MODEL_STATUS_CHANGED

logger = logging.getLogger(__name__)


class ModelService:
    def get_all_status(self) -> list[dict]:
        return model_manager.get_status()

    def get_model_status(self, model_key: str) -> Optional[dict]:
        statuses = model_manager.get_status(model_key)
        return statuses[0] if statuses else None

    async def set_model_enabled(self, model_key: str, enabled: bool) -> tuple[bool, str]:
        success, message = model_manager.set_enabled(model_key, enabled)
        if success:
            await kafka_service.publish(TOPIC_MODEL_STATUS_CHANGED, {
                "model_key": model_key,
                "enabled": enabled,
                "status": model_manager.get_status(model_key),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        return success, message

    async def activate_model_version(
        self, model_key: str, version_id: str, db: AsyncSession
    ) -> tuple[bool, str]:
        """Explicitly activate a trained model version. Does not auto-activate."""
        # Deactivate current active version for this model_key
        result = await db.execute(
            select(ModelVersion).where(
                ModelVersion.model_key == model_key,
                ModelVersion.is_active == True,
            )
        )
        current = result.scalar_one_or_none()
        if current:
            current.is_active = False

        # Activate the selected version
        result2 = await db.execute(
            select(ModelVersion).where(ModelVersion.version_id == version_id)
        )
        target = result2.scalar_one_or_none()
        if not target:
            return False, f"Version {version_id} not found"
        if target.model_key != model_key:
            return False, "Version does not belong to this model key"

        target.is_active = True
        target.activated_at = datetime.now(timezone.utc)
        await db.commit()

        logger.info(f"Model {model_key} version {version_id} activated")
        return True, "Activated"


model_service = ModelService()
