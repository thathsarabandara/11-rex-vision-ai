import logging
import uuid
from typing import Optional
from app.config.qdrant import get_qdrant_client
from app.config.settings import settings
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue, ScoredPoint

logger = logging.getLogger(__name__)


class VectorService:
    # ------------------------------------------------------------------
    # Face embeddings
    # ------------------------------------------------------------------
    async def upsert_face_embedding(
        self, point_id: str, vector: list[float], payload: dict
    ) -> None:
        client = get_qdrant_client()
        await client.upsert(
            collection_name=settings.QDRANT_FACE_COLLECTION,
            points=[PointStruct(id=point_id, vector=vector, payload=payload)],
        )

    async def search_face_embeddings(
        self, robot_id: str, query_vector: list[float], limit: int = 5
    ) -> list[dict]:
        client = get_qdrant_client()
        results = await client.search(
            collection_name=settings.QDRANT_FACE_COLLECTION,
            query_vector=query_vector,
            query_filter=Filter(
                must=[FieldCondition(key="robot_id", match=MatchValue(value=robot_id))]
            ),
            limit=limit,
            with_payload=True,
        )
        return [{"score": r.score, **r.payload} for r in results]

    async def search_all_face_embeddings(self, robot_id: str) -> list[dict]:
        """Scroll all active face embeddings for a robot (for pipeline use)."""
        client = get_qdrant_client()
        points, _ = await client.scroll(
            collection_name=settings.QDRANT_FACE_COLLECTION,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="robot_id", match=MatchValue(value=robot_id)),
                    FieldCondition(key="active", match=MatchValue(value=True)),
                ]
            ),
            with_vectors=True,
            with_payload=True,
            limit=500,
        )
        return [
            {
                "face_profile_id": p.payload.get("face_profile_id"),
                "display_name": p.payload.get("display_name"),
                "relationship": p.payload.get("relationship"),
                "embedding": p.vector,
            }
            for p in points
        ]

    async def delete_face_embedding(self, point_id: str) -> None:
        client = get_qdrant_client()
        await client.delete(
            collection_name=settings.QDRANT_FACE_COLLECTION,
            points_selector=[point_id],
        )

    # ------------------------------------------------------------------
    # Gesture embeddings
    # ------------------------------------------------------------------
    async def upsert_gesture_embedding(
        self, point_id: str, vector: list[float], payload: dict
    ) -> None:
        client = get_qdrant_client()
        await client.upsert(
            collection_name=settings.QDRANT_GESTURE_COLLECTION,
            points=[PointStruct(id=point_id, vector=vector, payload=payload)],
        )

    async def search_all_gesture_embeddings(self, robot_id: str) -> list[dict]:
        client = get_qdrant_client()
        points, _ = await client.scroll(
            collection_name=settings.QDRANT_GESTURE_COLLECTION,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="robot_id", match=MatchValue(value=robot_id)),
                    FieldCondition(key="active", match=MatchValue(value=True)),
                ]
            ),
            with_vectors=True,
            with_payload=True,
            limit=200,
        )
        return [
            {
                "gesture_profile_id": p.payload.get("gesture_profile_id"),
                "gesture_name": p.payload.get("gesture_name"),
                "gesture_type": p.payload.get("gesture_type"),
                "action_hint": p.payload.get("action_hint"),
                "vector": p.vector,
            }
            for p in points
        ]

    async def delete_gesture_embedding(self, point_id: str) -> None:
        client = get_qdrant_client()
        await client.delete(
            collection_name=settings.QDRANT_GESTURE_COLLECTION,
            points_selector=[point_id],
        )

    # ------------------------------------------------------------------
    # Scene / Visual memory
    # ------------------------------------------------------------------
    async def upsert_scene_embedding(
        self, point_id: str, vector: list[float], payload: dict
    ) -> None:
        client = get_qdrant_client()
        await client.upsert(
            collection_name=settings.QDRANT_SCENE_COLLECTION,
            points=[PointStruct(id=point_id, vector=vector, payload=payload)],
        )

    async def upsert_memory_embedding(
        self, point_id: str, vector: list[float], payload: dict
    ) -> None:
        client = get_qdrant_client()
        await client.upsert(
            collection_name=settings.QDRANT_MEMORY_COLLECTION,
            points=[PointStruct(id=point_id, vector=vector, payload=payload)],
        )

    async def search_memory(
        self, robot_id: str, query_vector: list[float], limit: int = 10
    ) -> list[dict]:
        client = get_qdrant_client()
        results = await client.search(
            collection_name=settings.QDRANT_MEMORY_COLLECTION,
            query_vector=query_vector,
            query_filter=Filter(
                must=[FieldCondition(key="robot_id", match=MatchValue(value=robot_id))]
            ),
            limit=limit,
            with_payload=True,
        )
        return [{"similarity": r.score, **r.payload} for r in results]


vector_service = VectorService()
