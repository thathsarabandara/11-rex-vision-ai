from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.config.database import get_db
from app.middleware.auth import get_current_user
from app.services.model_service import model_service
from app.services.ownership_service import ownership_service
from app.schemas.model import ModelStatusOut, ModelStateIn
from app.schemas.common import ResponseModel

router = APIRouter(tags=["Models"])


async def verify_robot_access(robot_id: str, user=Depends(get_current_user)):
    user_id = user.get("sub")
    if not await ownership_service.verify_ownership(user_id, robot_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    return user_id


@router.get("", response_model=ResponseModel[list])
async def list_models(robot_id: str, _=Depends(verify_robot_access)):
    return ResponseModel(success=True, data=model_service.get_all_status())


@router.get("/{model_key}", response_model=ResponseModel[dict])
async def get_model(robot_id: str, model_key: str, _=Depends(verify_robot_access)):
    info = model_service.get_model_status(model_key)
    if not info:
        raise HTTPException(status_code=404, detail=f"Model '{model_key}' not found")
    return ResponseModel(success=True, data=info)


@router.put("/{model_key}/state", response_model=ResponseModel[dict])
async def set_model_state(
    robot_id: str, model_key: str, body: ModelStateIn, _=Depends(verify_robot_access),
):
    success, message = await model_service.set_model_enabled(model_key, body.enabled)
    if not success:
        raise HTTPException(status_code=422, detail=message)
    return ResponseModel(success=True, data={"model_key": model_key, "enabled": body.enabled})


@router.post("/{model_key}/activate", response_model=ResponseModel[dict])
async def activate_model_version(
    robot_id: str, model_key: str, version_id: str,
    db: AsyncSession = Depends(get_db), _=Depends(verify_robot_access),
):
    """Explicitly activate a trained model version. Requires manual confirmation."""
    ok, message = await model_service.activate_model_version(model_key, version_id, db)
    if not ok:
        raise HTTPException(status_code=422, detail=message)
    return ResponseModel(success=True, data={"activated": True, "version_id": version_id})
