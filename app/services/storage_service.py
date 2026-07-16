import asyncio
import io
import logging
from typing import Optional
from app.config.storage import get_minio_client
from app.config.settings import settings

logger = logging.getLogger(__name__)


class StorageService:
    async def upload_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        """Upload bytes to MinIO and return the object key."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._sync_upload, key, data, content_type)
        return key

    def _sync_upload(self, key: str, data: bytes, content_type: str) -> None:
        client = get_minio_client()
        client.put_object(
            bucket_name=settings.MINIO_BUCKET,
            object_name=key,
            data=io.BytesIO(data),
            length=len(data),
            content_type=content_type,
        )

    async def get_presigned_url(self, key: str, expires_seconds: int = 3600) -> str:
        """Generate a presigned GET URL for the given object key."""
        from datetime import timedelta
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_presign, key, expires_seconds)

    def _sync_presign(self, key: str, expires_seconds: int) -> str:
        from datetime import timedelta
        client = get_minio_client()
        return client.presigned_get_object(
            bucket_name=settings.MINIO_BUCKET,
            object_name=key,
            expires=timedelta(seconds=expires_seconds),
        )

    async def delete_prefix(self, prefix: str) -> None:
        """Delete all objects under a MinIO prefix."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._sync_delete_prefix, prefix)

    def _sync_delete_prefix(self, prefix: str) -> None:
        client = get_minio_client()
        objects = client.list_objects(settings.MINIO_BUCKET, prefix=prefix, recursive=True)
        for obj in objects:
            client.remove_object(settings.MINIO_BUCKET, obj.object_name)

    async def delete_object(self, key: str) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: get_minio_client().remove_object(settings.MINIO_BUCKET, key))


storage_service = StorageService()
