from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.config.database import get_db
from app.middleware.auth import get_current_user
from app.services.tracking_service import tracking_service
from app.services.ownership_service import ownership_service
from app.schemas.tracking import TrackingTargetIn, TrackingTargetOut
from app.schemas.common import ResponseModel

router = APIRouter(tags=["Tracking"])


async def verify_robot_access(robot_id: str, user=Depends(get_current_user)):
    user_id = user.get("sub")
    if not await ownership_service.verify_ownership(user_id, robot_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    return user_id


@router.post("", response_model=ResponseModel[TrackingTargetOut])
async def set_tracking_target(
    robot_id: str, body: TrackingTargetIn, _=Depends(verify_robot_access),
):
    await tracking_service.set_target(robot_id, body.track_id)
    return ResponseModel(success=True, data=TrackingTargetOut(
        track_id=body.track_id, robot_id=robot_id, active=True,
    ))


@router.delete("", response_model=ResponseModel[dict])
async def clear_tracking_target(robot_id: str, _=Depends(verify_robot_access)):
    await tracking_service.clear_target(robot_id)
    return ResponseModel(success=True, data={"active": False})


@router.get("", response_model=ResponseModel[TrackingTargetOut])
async def get_tracking_target(robot_id: str, _=Depends(verify_robot_access)):
    target = await tracking_service.get_target(robot_id)
    if not target:
        return ResponseModel(success=True, data=TrackingTargetOut(track_id=None, robot_id=robot_id, active=False))
    return ResponseModel(success=True, data=TrackingTargetOut(
        track_id=target.get("track_id"), robot_id=robot_id, active=target.get("active", False),
    ))
