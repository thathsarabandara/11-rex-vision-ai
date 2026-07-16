from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.config.database import get_db
from app.middleware.auth import get_current_user
from app.services.training_service import training_service
from app.services.ownership_service import ownership_service
from app.schemas.training import TrainingJobCreateIn, TrainingJobOut
from app.schemas.common import ResponseModel

router = APIRouter(tags=["Training"])


async def verify_robot_access(robot_id: str, user=Depends(get_current_user)):
    user_id = user.get("sub")
    if not await ownership_service.verify_ownership(user_id, robot_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    return user_id


@router.post("", response_model=ResponseModel[TrainingJobOut], status_code=202)
async def create_training_job(
    robot_id: str, body: TrainingJobCreateIn,
    db: AsyncSession = Depends(get_db),
    user=Depends(get_current_user),
):
    user_id = user.get("sub")
    if not await ownership_service.verify_ownership(user_id, robot_id):
        raise HTTPException(status_code=403, detail="Access denied")
    try:
        job = await training_service.create_job(
            robot_id=robot_id, user_id=user_id,
            training_type=body.training_type,
            dataset_id=body.dataset_id,
            base_model=body.base_model,
            epochs=body.epochs,
            image_size=body.image_size or 640,
            db=db,
        )
        return ResponseModel(success=True, data=TrainingJobOut.model_validate(job))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.get("", response_model=ResponseModel[list])
async def list_training_jobs(robot_id: str, db: AsyncSession = Depends(get_db), _=Depends(verify_robot_access)):
    jobs = await training_service.list_jobs(robot_id, db)
    return ResponseModel(success=True, data=[TrainingJobOut.model_validate(j).model_dump() for j in jobs])


@router.get("/{job_id}", response_model=ResponseModel[TrainingJobOut])
async def get_training_job(robot_id: str, job_id: str, db: AsyncSession = Depends(get_db), _=Depends(verify_robot_access)):
    job = await training_service.get_job(robot_id, job_id, db)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    return ResponseModel(success=True, data=TrainingJobOut.model_validate(job))


@router.post("/{job_id}/cancel", response_model=ResponseModel[dict])
async def cancel_training_job(
    robot_id: str, job_id: str,
    db: AsyncSession = Depends(get_db), _=Depends(verify_robot_access),
):
    cancelled = await training_service.cancel_job(robot_id, job_id, db)
    if not cancelled:
        raise HTTPException(status_code=422, detail="Job cannot be cancelled")
    return ResponseModel(success=True, data={"cancelled": True})
