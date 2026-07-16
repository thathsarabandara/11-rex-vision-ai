from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
from app.config.database import get_db
from app.middleware.auth import get_current_user
from app.services.face_service import face_service
from app.services.ownership_service import ownership_service
from app.schemas.face import FaceRegistrationIn, FaceProfileOut, FaceProfilePatch
from app.schemas.common import ResponseModel

router = APIRouter(tags=["Faces"])


async def verify_robot_access(robot_id: str, user=Depends(get_current_user)):
    user_id = user.get("sub")
    if not await ownership_service.verify_ownership(user_id, robot_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"success": False, "error": {"code": "ROBOT_NOT_OWNED", "message": "Access denied"}},
        )
    return user_id


@router.post("", response_model=ResponseModel[FaceProfileOut], status_code=201)
async def register_face(
    robot_id: str,
    display_name: str = Form(...),
    relationship: str = Form(default="GUEST"),
    images: List[UploadFile] = File(...),
    db: AsyncSession = Depends(get_db),
    user=Depends(get_current_user),
):
    user_id = user.get("sub")
    if not await ownership_service.verify_ownership(user_id, robot_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    if not images:
        raise HTTPException(status_code=422, detail="At least one image required")

    image_data = [await img.read() for img in images]
    content_types = [img.content_type or "image/jpeg" for img in images]

    try:
        profile = await face_service.register_face(
            robot_id=robot_id, user_id=user_id,
            display_name=display_name, relationship=relationship,
            image_bytes_list=image_data, content_types=content_types, db=db,
        )
        return ResponseModel(success=True, data=FaceProfileOut.model_validate(profile))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.get("", response_model=ResponseModel[list])
async def list_faces(robot_id: str, db: AsyncSession = Depends(get_db), _=Depends(verify_robot_access)):
    profiles = await face_service.list_profiles(robot_id, db)
    return ResponseModel(success=True, data=[FaceProfileOut.model_validate(p).model_dump() for p in profiles])


@router.get("/{face_profile_id}", response_model=ResponseModel[FaceProfileOut])
async def get_face(robot_id: str, face_profile_id: str, db: AsyncSession = Depends(get_db), _=Depends(verify_robot_access)):
    profile = await face_service.get_profile(robot_id, face_profile_id, db)
    if not profile:
        raise HTTPException(status_code=404, detail="Face profile not found")
    return ResponseModel(success=True, data=FaceProfileOut.model_validate(profile))


@router.patch("/{face_profile_id}", response_model=ResponseModel[FaceProfileOut])
async def update_face(
    robot_id: str, face_profile_id: str, body: FaceProfilePatch,
    db: AsyncSession = Depends(get_db), _=Depends(verify_robot_access),
):
    profile = await face_service.update_profile(robot_id, face_profile_id, body.model_dump(exclude_none=True), db)
    if not profile:
        raise HTTPException(status_code=404, detail="Face profile not found")
    return ResponseModel(success=True, data=FaceProfileOut.model_validate(profile))


@router.delete("/{face_profile_id}", response_model=ResponseModel[dict])
async def delete_face(
    robot_id: str, face_profile_id: str,
    db: AsyncSession = Depends(get_db), _=Depends(verify_robot_access),
):
    deleted = await face_service.delete_profile(robot_id, face_profile_id, db)
    if not deleted:
        raise HTTPException(status_code=404, detail="Face profile not found")
    return ResponseModel(success=True, data={"deleted": True})


@router.post("/{face_profile_id}/samples", response_model=ResponseModel[FaceProfileOut])
async def add_face_samples(
    robot_id: str, face_profile_id: str,
    images: List[UploadFile] = File(...),
    db: AsyncSession = Depends(get_db),
    user=Depends(get_current_user),
):
    """Add more sample images to an existing face profile."""
    user_id = user.get("sub")
    if not await ownership_service.verify_ownership(user_id, robot_id):
        raise HTTPException(status_code=403, detail="Access denied")

    profile = await face_service.get_profile(robot_id, face_profile_id, db)
    if not profile:
        raise HTTPException(status_code=404, detail="Face profile not found")

    image_data = [await img.read() for img in images]
    content_types = [img.content_type or "image/jpeg" for img in images]

    try:
        # Re-register with additional samples (updates embedding)
        updated = await face_service.register_face(
            robot_id=robot_id, user_id=user_id,
            display_name=profile.display_name, relationship=profile.relationship,
            image_bytes_list=image_data, content_types=content_types, db=db,
        )
        return ResponseModel(success=True, data=FaceProfileOut.model_validate(updated))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
