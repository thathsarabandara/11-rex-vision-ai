from fastapi import APIRouter, Depends, HTTPException, status
from app.middleware.auth import get_current_user
from app.services.vector_service import vector_service
from app.services.ownership_service import ownership_service
from app.schemas.memory import MemorySearchIn, MemorySearchResult
from app.schemas.common import ResponseModel

router = APIRouter(tags=["Memory"])


async def verify_robot_access(robot_id: str, user=Depends(get_current_user)):
    user_id = user.get("sub")
    if not await ownership_service.verify_ownership(user_id, robot_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    return user_id


@router.post("/search", response_model=ResponseModel[list])
async def search_visual_memory(
    robot_id: str, body: MemorySearchIn, _=Depends(verify_robot_access),
):
    """Search visual memory using a text query embedded via a simple encoding."""
    # Create a simple query vector from text (placeholder — replace with CLIP/sentence encoder)
    import hashlib
    import numpy as np
    from app.config.qdrant import MEMORY_VECTOR_SIZE

    # Deterministic pseudo-vector from query hash (functional stub)
    h = hashlib.sha256(body.query.encode()).digest()
    rng = np.random.default_rng(int.from_bytes(h[:4], "big"))
    query_vector = rng.standard_normal(MEMORY_VECTOR_SIZE).tolist()

    results = await vector_service.search_memory(robot_id, query_vector, limit=body.limit)

    out = [
        MemorySearchResult(
            scene_id=r.get("scene_id", ""),
            description=r.get("description", ""),
            captured_at=r.get("captured_at", ""),
            similarity=round(r.get("similarity", 0.0), 4),
            media_url=r.get("media_url"),
        ).model_dump()
        for r in results
    ]
    return ResponseModel(success=True, data=out)
