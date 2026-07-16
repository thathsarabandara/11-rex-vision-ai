import logging
from typing import Optional
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams
from app.config.settings import settings

logger = logging.getLogger(__name__)

_qdrant_client: Optional[AsyncQdrantClient] = None

FACE_VECTOR_SIZE = 512
GESTURE_VECTOR_SIZE = 63  # 21 landmarks * 3 (x, y, z)
SCENE_VECTOR_SIZE = 512
MEMORY_VECTOR_SIZE = 512

COLLECTIONS = {
    settings.QDRANT_FACE_COLLECTION: FACE_VECTOR_SIZE,
    settings.QDRANT_GESTURE_COLLECTION: GESTURE_VECTOR_SIZE,
    settings.QDRANT_SCENE_COLLECTION: SCENE_VECTOR_SIZE,
    settings.QDRANT_MEMORY_COLLECTION: MEMORY_VECTOR_SIZE,
}


def get_qdrant_client() -> AsyncQdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        kwargs: dict = {"url": settings.QDRANT_URL}
        if settings.QDRANT_API_KEY:
            kwargs["api_key"] = settings.QDRANT_API_KEY
        _qdrant_client = AsyncQdrantClient(**kwargs)
    return _qdrant_client


async def ensure_collections() -> None:
    """Create Qdrant collections if they do not exist."""
    client = get_qdrant_client()
    existing = {c.name for c in await client.get_collections()}
    for name, size in COLLECTIONS.items():
        if name not in existing:
            await client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=size, distance=Distance.COSINE),
            )
            logger.info(f"Created Qdrant collection: {name}")
