import logging
import uuid
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.models.gesture_profile import GestureProfile
from app.services.vector_service import vector_service
from app.ai.hand_landmarker import hand_landmarker

logger = logging.getLogger(__name__)

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024


class GestureService:
    async def register_gesture(
        self,
        robot_id: str,
        user_id: str,
        gesture_name: str,
        description: Optional[str],
        gesture_type: str,
        action_hint: Optional[str],
        sample_images: list[bytes],
        content_types: list[str],
        db: AsyncSession,
    ) -> GestureProfile:
        for ct, img in zip(content_types, sample_images):
            if ct not in ALLOWED_IMAGE_TYPES:
                raise ValueError(f"Unsupported image type: {ct}")

        # Extract landmark vectors from each sample image
        vectors = []
        for img_bytes in sample_images:
            vecs = self._extract_vectors(img_bytes)
            vectors.extend(vecs)

        if not vectors:
            raise ValueError("No valid hand landmarks detected in sample images")

        gesture_profile_id = str(uuid.uuid4())
        qdrant_ids: list[str] = []

        # Average vector as representative embedding
        import numpy as np
        avg_vector = list(np.mean(vectors, axis=0))

        point_id = str(uuid.uuid4())
        await vector_service.upsert_gesture_embedding(
            point_id=point_id,
            vector=avg_vector,
            payload={
                "gesture_profile_id": gesture_profile_id,
                "robot_id": robot_id,
                "gesture_name": gesture_name,
                "gesture_type": gesture_type,
                "active": True,
            },
        )
        qdrant_ids.append(point_id)

        profile = GestureProfile(
            gesture_profile_id=gesture_profile_id,
            robot_id=robot_id,
            user_id=user_id,
            gesture_name=gesture_name,
            description=description,
            gesture_type=gesture_type,
            action_hint=action_hint,
            sample_count=len(sample_images),
            qdrant_point_ids=",".join(qdrant_ids),
        )
        db.add(profile)
        await db.commit()
        await db.refresh(profile)
        return profile

    def _extract_vectors(self, img_bytes: bytes) -> list[list[float]]:
        try:
            import numpy as np
            import cv2  # type: ignore
            arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                return []
            hands = hand_landmarker.extract(frame)
            return [h["vector"] for h in hands if h.get("vector")]
        except Exception as exc:
            logger.error(f"Gesture vector extraction error: {exc}")
            return []

    async def list_profiles(self, robot_id: str, db: AsyncSession) -> list[GestureProfile]:
        result = await db.execute(
            select(GestureProfile).where(GestureProfile.robot_id == robot_id)
        )
        return list(result.scalars().all())

    async def get_profile(self, robot_id: str, gesture_id: str, db: AsyncSession) -> Optional[GestureProfile]:
        result = await db.execute(
            select(GestureProfile).where(
                GestureProfile.robot_id == robot_id,
                GestureProfile.gesture_profile_id == gesture_id,
            )
        )
        return result.scalar_one_or_none()

    async def update_profile(
        self, robot_id: str, gesture_id: str, updates: dict, db: AsyncSession
    ) -> Optional[GestureProfile]:
        profile = await self.get_profile(robot_id, gesture_id, db)
        if not profile:
            return None
        for field, value in updates.items():
            if value is not None and hasattr(profile, field):
                setattr(profile, field, value)
        await db.commit()
        await db.refresh(profile)
        return profile

    async def delete_profile(self, robot_id: str, gesture_id: str, db: AsyncSession) -> bool:
        profile = await self.get_profile(robot_id, gesture_id, db)
        if not profile:
            return False
        if profile.qdrant_point_ids:
            for pid in profile.qdrant_point_ids.split(","):
                await vector_service.delete_gesture_embedding(pid.strip())
        await db.delete(profile)
        await db.commit()
        return True

    async def get_active_profiles(self, robot_id: str) -> list[dict]:
        """Return gesture profiles with vectors for pipeline matching."""
        return await vector_service.search_all_gesture_embeddings(robot_id)


gesture_service = GestureService()
