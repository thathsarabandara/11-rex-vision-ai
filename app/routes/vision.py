from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from typing import Optional
from app.config.database import get_db
from app.middleware.auth import get_current_user
from app.services.vision_service import vision_service
from app.services.ownership_service import ownership_service
from app.models.vision_event import VisionEvent
from app.schemas.vision import FeatureFlagsIn, FeatureFlagsOut, CameraSourceIn, CameraSourceOut, VisionEventOut
from app.schemas.common import ResponseModel

router = APIRouter(tags=["Vision"])


async def verify_robot_access(robot_id: str, user=Depends(get_current_user)):
    user_id = user.get("sub")
    if not await ownership_service.verify_ownership(user_id, robot_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"success": False, "error": {"code": "ROBOT_NOT_OWNED", "message": "Access denied"}},
        )
    return user_id


@router.get("/latest", response_model=ResponseModel[dict])
async def get_latest_vision_state(robot_id: str, _=Depends(verify_robot_access)):
    state = await vision_service.get_latest_state(robot_id)
    if not state:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail={"success": False, "error": {"code": "NO_STATE", "message": "No vision state available"}})
    return ResponseModel(success=True, data=state)


@router.get("/objects", response_model=ResponseModel[list])
async def get_objects(robot_id: str, _=Depends(verify_robot_access)):
    state = await vision_service.get_latest_state(robot_id)
    return ResponseModel(success=True, data=(state or {}).get("detections", []))


@router.get("/people", response_model=ResponseModel[list])
async def get_people(robot_id: str, _=Depends(verify_robot_access)):
    state = await vision_service.get_latest_state(robot_id)
    detections = (state or {}).get("detections", [])
    people = [d for d in detections if d.get("class_name") == "person"]
    return ResponseModel(success=True, data=people)


@router.get("/scene", response_model=ResponseModel[dict])
async def get_scene(robot_id: str, _=Depends(verify_robot_access)):
    state = await vision_service.get_latest_state(robot_id)
    return ResponseModel(success=True, data=(state or {}).get("scene", {}))


@router.get("/performance", response_model=ResponseModel[dict])
async def get_performance(robot_id: str, _=Depends(verify_robot_access)):
    state = await vision_service.get_latest_state(robot_id)
    return ResponseModel(success=True, data=(state or {}).get("performance", {}))


@router.get("/events", response_model=ResponseModel[list])
async def get_vision_events(
    robot_id: str,
    event_type: Optional[str] = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    _=Depends(verify_robot_access),
):
    q = select(VisionEvent).where(VisionEvent.robot_id == robot_id).order_by(VisionEvent.occurred_at.desc())
    if event_type:
        q = q.where(VisionEvent.event_type == event_type)
    q = q.limit(limit).offset(offset)
    result = await db.execute(q)
    events = result.scalars().all()
    return ResponseModel(success=True, data=[
        {"event_id": e.event_id, "event_type": e.event_type, "severity": e.severity,
         "data": e.data, "occurred_at": e.occurred_at.isoformat()} for e in events
    ])


@router.get("/features", response_model=ResponseModel[FeatureFlagsOut])
async def get_features(robot_id: str, db: AsyncSession = Depends(get_db), _=Depends(verify_robot_access)):
    features = await vision_service.get_features(robot_id, db)
    return ResponseModel(success=True, data=FeatureFlagsOut(**features))


@router.put("/features", response_model=ResponseModel[FeatureFlagsOut])
async def update_features(
    robot_id: str,
    body: FeatureFlagsIn,
    db: AsyncSession = Depends(get_db),
    _=Depends(verify_robot_access),
):
    try:
        updated = await vision_service.update_features(robot_id, body.model_dump(exclude_none=True), db)
        return ResponseModel(success=True, data=FeatureFlagsOut(**updated))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))


@router.post("/cameras", response_model=ResponseModel[CameraSourceOut])
async def create_camera_source(
    robot_id: str,
    body: CameraSourceIn,
    db: AsyncSession = Depends(get_db),
    _=Depends(verify_robot_access),
):
    source = await vision_service.create_camera_source(robot_id, body.model_dump(), db)
    return ResponseModel(success=True, data=CameraSourceOut.model_validate(source))


@router.get("/cameras", response_model=ResponseModel[list])
async def list_camera_sources(robot_id: str, db: AsyncSession = Depends(get_db), _=Depends(verify_robot_access)):
    sources = await vision_service.list_camera_sources(robot_id, db)
    return ResponseModel(success=True, data=[CameraSourceOut.model_validate(s).model_dump() for s in sources])
