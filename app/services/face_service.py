import logging
import uuid
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.models.face_profile import FaceProfile
from app.services.vector_service import vector_service
from app.services.storage_service import storage_service
from app.ai.face_recognizer import face_recognizer
from app.config.settings import settings

logger = logging.getLogger(__name__)

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB


class FaceService:
    async def register_face(
        self,
        robot_id: str,
        user_id: str,
        display_name: str,
        relationship: str,
        image_bytes_list: list[bytes],
        content_types: list[str],
        db: AsyncSession,
    ) -> FaceProfile:
        # Validate images
        for ct, img in zip(content_types, image_bytes_list):
            if ct not in ALLOWED_IMAGE_TYPES:
                raise ValueError(f"Unsupported image type: {ct}")
            if len(img) > MAX_IMAGE_BYTES:
                raise ValueError("Image too large (max 10 MB)")

        # Extract embeddings from each image
        embeddings = []
        for img_bytes in image_bytes_list:
            emb = self._extract_embedding(img_bytes)
            if emb:
                embeddings.append(emb)

        if not embeddings:
            raise ValueError("No usable face detected in provided images")

        # Average embeddings
        import numpy as np
        avg_emb = list(np.mean(embeddings, axis=0))
        avg_emb = face_recognizer.normalize_embedding(avg_emb)

        # Create face profile
        face_profile_id = str(uuid.uuid4())
        qdrant_point_id = str(uuid.uuid4())
        minio_prefix = f"faces/{robot_id}/{face_profile_id}/"

        # Store embedding in Qdrant
        await vector_service.upsert_face_embedding(
            point_id=qdrant_point_id,
            vector=avg_emb,
            payload={
                "face_profile_id": face_profile_id,
                "robot_id": robot_id,
                "display_name": display_name,
                "relationship": relationship,
                "active": True,
            },
        )

        # Store reference images in MinIO
        for i, img_bytes in enumerate(image_bytes_list):
            key = f"{minio_prefix}sample_{i}.jpg"
            await storage_service.upload_bytes(key, img_bytes, content_type="image/jpeg")

        # Persist profile in MySQL
        profile = FaceProfile(
            face_profile_id=face_profile_id,
            robot_id=robot_id,
            user_id=user_id,
            display_name=display_name,
            relationship=relationship,
            sample_count=len(image_bytes_list),
            qdrant_point_id=qdrant_point_id,
            minio_prefix=minio_prefix,
        )
        db.add(profile)
        await db.commit()
        await db.refresh(profile)
        return profile

    def _extract_embedding(self, img_bytes: bytes) -> Optional[list[float]]:
        """Extract face embedding using InsightFace."""
        try:
            import numpy as np
            import cv2  # type: ignore
            arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                return None
            from app.ai.face_detector import face_detector
            faces = face_detector.detect(frame)
            if not faces:
                return None
            # Take the highest-confidence face
            best = max(faces, key=lambda f: f.get("det_score", 0))
            return best.get("embedding")
        except Exception as exc:
            logger.error(f"Embedding extraction error: {exc}")
            return None

    async def list_profiles(self, robot_id: str, db: AsyncSession) -> list[FaceProfile]:
        result = await db.execute(
            select(FaceProfile).where(FaceProfile.robot_id == robot_id)
        )
        return list(result.scalars().all())

    async def get_profile(self, robot_id: str, face_profile_id: str, db: AsyncSession) -> Optional[FaceProfile]:
        result = await db.execute(
            select(FaceProfile).where(
                FaceProfile.robot_id == robot_id,
                FaceProfile.face_profile_id == face_profile_id,
            )
        )
        return result.scalar_one_or_none()

    async def update_profile(
        self, robot_id: str, face_profile_id: str, updates: dict, db: AsyncSession
    ) -> Optional[FaceProfile]:
        profile = await self.get_profile(robot_id, face_profile_id, db)
        if not profile:
            return None
        for field, value in updates.items():
            if value is not None and hasattr(profile, field):
                setattr(profile, field, value)
        await db.commit()
        await db.refresh(profile)
        return profile

    async def delete_profile(self, robot_id: str, face_profile_id: str, db: AsyncSession) -> bool:
        profile = await self.get_profile(robot_id, face_profile_id, db)
        if not profile:
            return False

        # Remove Qdrant embedding
        if profile.qdrant_point_id:
            await vector_service.delete_face_embedding(profile.qdrant_point_id)

        # Remove MinIO images
        if profile.minio_prefix:
            await storage_service.delete_prefix(profile.minio_prefix)

        await db.delete(profile)
        await db.commit()
        return True

    async def get_active_embeddings(self, robot_id: str) -> list[dict]:
        """Retrieve face embeddings for all active profiles from Qdrant."""
        return await vector_service.search_all_face_embeddings(robot_id)


face_service = FaceService()
