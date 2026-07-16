import logging
from minio import Minio
from app.config.settings import settings

logger = logging.getLogger(__name__)

_minio_client = None


def get_minio_client() -> Minio:
    global _minio_client
    if _minio_client is None:
        _minio_client = Minio(
            endpoint=settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=settings.MINIO_SECURE,
        )
    return _minio_client


def ensure_bucket() -> None:
    """Create the MinIO bucket if it does not exist."""
    client = get_minio_client()
    if not client.bucket_exists(settings.MINIO_BUCKET):
        client.make_bucket(settings.MINIO_BUCKET)
        logger.info(f"Created MinIO bucket: {settings.MINIO_BUCKET}")
