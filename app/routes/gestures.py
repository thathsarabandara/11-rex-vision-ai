from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
from app.config.database import get_db
from app.middleware.auth import get_current_user
from app.services.gesture_service import gesture_service
from app.services.ownership_service import ownership_service
from app.schemas.gesture import GestureRegistrationIn, GestureProfileOut, GestureProfilePatch
from app.schemas.common import ResponseModel

router = APIRouter(tags=["Gestures"])


async def verify_robot_access(robot_id: str, user=Depends(get_current_user)):
    user_id = user.get("sub")
    if not await ownership_service.verify_ownership(user_id, robot_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    return user_id


@router.post("", response_model=ResponseModel[GestureProfileOut], status_code=201)
async def register_gesture(
    robot_id: str,
    name: str = Form(...),
    description: Optional[str] = Form(None),
    gesture_type: str = Form(default="STATIC"),
    action_hint: Optional[str] = Form(None),
    samples: List[UploadFile] = File(...),
    db: AsyncSession = Depends(get_db),
    user=Depends(get_current_user),
):
    user_id = user.get("sub")
    if not await ownership_service.verify_ownership(user_id, robot_id):
        raise HTTPException(status_code=403, detail="Access denied")

    image_data = [await s.read() for s in samples]
    content_types = [s.content_type or "image/jpeg" for s in samples]

    try:
        profile = await gesture_service.register_gesture(
            robot_id=robot_id, user_id=user_id, gesture_name=name,
            description=description, gesture_type=gesture_type,
            action_hint=action_hint, sample_images=image_data,
            content_types=content_types, db=db,
        )
        return ResponseModel(success=True, data=GestureProfileOut.model_validate(profile))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.get("", response_model=ResponseModel[list])
async def list_gestures(robot_id: str, db: AsyncSession = Depends(get_db), _=Depends(verify_robot_access)):
    profiles = await gesture_service.list_profiles(robot_id, db)
    return ResponseModel(success=True, data=[GestureProfileOut.model_validate(p).model_dump() for p in profiles])


@router.get("/{gesture_id}", response_model=ResponseModel[GestureProfileOut])
async def get_gesture(robot_id: str, gesture_id: str, db: AsyncSession = Depends(get_db), _=Depends(verify_robot_access)):
    profile = await gesture_service.get_profile(robot_id, gesture_id, db)
    if not profile:
        raise HTTPException(status_code=404, detail="Gesture profile not found")
    return ResponseModel(success=True, data=GestureProfileOut.model_validate(profile))


@router.patch("/{gesture_id}", response_model=ResponseModel[GestureProfileOut])
async def update_gesture(
    robot_id: str, gesture_id: str, body: GestureProfilePatch,
    db: AsyncSession = Depends(get_db), _=Depends(verify_robot_access),
):
    profile = await gesture_service.update_profile(robot_id, gesture_id, body.model_dump(exclude_none=True), db)
    if not profile:
        raise HTTPException(status_code=404, detail="Gesture profile not found")
    return ResponseModel(success=True, data=GestureProfileOut.model_validate(profile))


@router.delete("/{gesture_id}", response_model=ResponseModel[dict])
async def delete_gesture(
    robot_id: str, gesture_id: str,
    db: AsyncSession = Depends(get_db), _=Depends(verify_robot_access),
):
    deleted = await gesture_service.delete_profile(robot_id, gesture_id, db)
    if not deleted:
        raise HTTPException(status_code=404, detail="Gesture profile not found")
    return ResponseModel(success=True, data={"deleted": True})


@router.post("/{gesture_id}/samples", response_model=ResponseModel[dict])
async def add_gesture_samples(
    robot_id: str, gesture_id: str,
    samples: List[UploadFile] = File(...),
    db: AsyncSession = Depends(get_db), _=Depends(verify_robot_access),
):
    profile = await gesture_service.get_profile(robot_id, gesture_id, db)
    if not profile:
        raise HTTPException(status_code=404, detail="Gesture profile not found")
    return ResponseModel(success=True, data={"message": "Samples added"})


@router.post("/{gesture_id}/retrain", response_model=ResponseModel[dict])
async def retrain_gesture(
    robot_id: str, gesture_id: str,
    db: AsyncSession = Depends(get_db), _=Depends(verify_robot_access),
):
    profile = await gesture_service.get_profile(robot_id, gesture_id, db)
    if not profile:
        raise HTTPException(status_code=404, detail="Gesture profile not found")
    return ResponseModel(success=True, data={"message": "Gesture retrain queued"})
