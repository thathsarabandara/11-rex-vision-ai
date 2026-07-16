import json
import logging
from datetime import datetime, timezone
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.config.redis import get_redis
from app.config.settings import settings
from app.models.vision_configuration import VisionConfiguration
from app.models.camera_source import CameraSource

logger = logging.getLogger(__name__)

DEFAULT_FEATURES = {
    "object_detection": settings.OBJECT_DETECTION_ENABLED,
    "face_detection": settings.FACE_DETECTION_ENABLED,
    "face_recognition": settings.FACE_RECOGNITION_ENABLED,
    "expression_estimation": settings.EXPRESSION_ESTIMATION_ENABLED,
    "person_tracking": settings.PERSON_TRACKING_ENABLED,
    "gesture_detection": settings.GESTURE_DETECTION_ENABLED,
    "scene_understanding": settings.SCENE_UNDERSTANDING_ENABLED,
    "scene_description": settings.SCENE_DESCRIPTION_ENABLED,
    "visual_obstacle_detection": settings.VISUAL_OBSTACLE_ENABLED,
    "low_light_detection": settings.LOW_LIGHT_DETECTION_ENABLED,
    "incident_capture": settings.INCIDENT_CAPTURE_ENABLED,
}

FEATURE_DEPENDENCIES: dict[str, list[str]] = {
    "face_recognition": ["face_detection"],
    "expression_estimation": ["face_detection"],
    "person_tracking": ["object_detection"],
    "scene_description": ["scene_understanding"],
}


def validate_feature_dependencies(features: dict) -> Optional[str]:
    """Return an error message if a dependency is violated, else None."""
    for feature, deps in FEATURE_DEPENDENCIES.items():
        if features.get(feature):
            for dep in deps:
                if not features.get(dep):
                    return f"Feature '{feature}' requires '{dep}' to be enabled."
    return None


class VisionService:
    async def get_or_create_config(self, robot_id: str, db: AsyncSession) -> VisionConfiguration:
        result = await db.execute(
            select(VisionConfiguration).where(VisionConfiguration.robot_id == robot_id)
        )
        config = result.scalar_one_or_none()
        if config is None:
            config = VisionConfiguration(
                robot_id=robot_id,
                features=DEFAULT_FEATURES.copy(),
                thresholds={
                    "face_known": settings.FACE_KNOWN_THRESHOLD,
                    "face_uncertain": settings.FACE_UNCERTAIN_THRESHOLD,
                    "object_confidence": settings.OBJECT_CONFIDENCE_THRESHOLD,
                    "gesture_confidence": settings.GESTURE_CONFIDENCE_THRESHOLD,
                },
            )
            db.add(config)
            await db.commit()
            await db.refresh(config)
        return config

    async def get_features(self, robot_id: str, db: AsyncSession) -> dict:
        # Try Redis cache first
        redis = await get_redis()
        cached = await redis.get(f"rex:vision:features:{robot_id}")
        if cached:
            return json.loads(cached)

        config = await self.get_or_create_config(robot_id, db)
        features = config.features or DEFAULT_FEATURES.copy()
        await redis.set(f"rex:vision:features:{robot_id}", json.dumps(features), ex=60)
        return features

    async def update_features(self, robot_id: str, updates: dict, db: AsyncSession) -> dict:
        config = await self.get_or_create_config(robot_id, db)
        current = config.features or DEFAULT_FEATURES.copy()
        merged = {**current, **{k: v for k, v in updates.items() if v is not None}}

        error = validate_feature_dependencies(merged)
        if error:
            raise ValueError(error)

        config.features = merged
        await db.commit()
        await db.refresh(config)

        # Invalidate cache
        redis = await get_redis()
        await redis.delete(f"rex:vision:features:{robot_id}")
        return merged

    async def get_latest_state(self, robot_id: str) -> Optional[dict]:
        redis = await get_redis()
        raw = await redis.get(f"rex:vision:latest:{robot_id}")
        return json.loads(raw) if raw else None

    async def set_latest_state(self, robot_id: str, state: dict) -> None:
        redis = await get_redis()
        await redis.set(f"rex:vision:latest:{robot_id}", json.dumps(state), ex=30)

    # Camera sources
    async def create_camera_source(self, robot_id: str, data: dict, db: AsyncSession) -> CameraSource:
        source = CameraSource(robot_id=robot_id, **data)
        db.add(source)
        await db.commit()
        await db.refresh(source)
        return source

    async def list_camera_sources(self, robot_id: str, db: AsyncSession) -> list[CameraSource]:
        result = await db.execute(
            select(CameraSource).where(CameraSource.robot_id == robot_id)
        )
        return list(result.scalars().all())


vision_service = VisionService()
