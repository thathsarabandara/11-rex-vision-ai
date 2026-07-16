import uuid
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.config.database import get_db
from app.middleware.auth import get_current_user
from app.services.ownership_service import ownership_service
from app.models.vision_dataset import VisionDataset
from app.schemas.dataset import DatasetCreateIn, DatasetOut
from app.schemas.common import ResponseModel
from app.config.settings import settings

router = APIRouter(tags=["Datasets"])


async def verify_robot_access(robot_id: str, user=Depends(get_current_user)):
    user_id = user.get("sub")
    if not await ownership_service.verify_ownership(user_id, robot_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    return user_id


@router.post("", response_model=ResponseModel[DatasetOut], status_code=201)
async def create_dataset(
    robot_id: str, body: DatasetCreateIn,
    db: AsyncSession = Depends(get_db),
    user=Depends(get_current_user),
):
    user_id = user.get("sub")
    if not await ownership_service.verify_ownership(user_id, robot_id):
        raise HTTPException(status_code=403, detail="Access denied")

    dataset_id = str(uuid.uuid4())
    minio_prefix = f"{settings.DATASET_PREFIX}{robot_id}/{dataset_id}/"

    ds = VisionDataset(
        dataset_id=dataset_id, robot_id=robot_id, user_id=user_id,
        name=body.name, description=body.description,
        dataset_type=body.dataset_type, minio_prefix=minio_prefix,
    )
    db.add(ds)
    await db.commit()
    await db.refresh(ds)
    return ResponseModel(success=True, data=DatasetOut.model_validate(ds))


@router.get("", response_model=ResponseModel[list])
async def list_datasets(robot_id: str, db: AsyncSession = Depends(get_db), _=Depends(verify_robot_access)):
    result = await db.execute(select(VisionDataset).where(VisionDataset.robot_id == robot_id))
    datasets = result.scalars().all()
    return ResponseModel(success=True, data=[DatasetOut.model_validate(d).model_dump() for d in datasets])


@router.get("/{dataset_id}", response_model=ResponseModel[DatasetOut])
async def get_dataset(robot_id: str, dataset_id: str, db: AsyncSession = Depends(get_db), _=Depends(verify_robot_access)):
    result = await db.execute(
        select(VisionDataset).where(VisionDataset.robot_id == robot_id, VisionDataset.dataset_id == dataset_id)
    )
    ds = result.scalar_one_or_none()
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return ResponseModel(success=True, data=DatasetOut.model_validate(ds))


@router.delete("/{dataset_id}", response_model=ResponseModel[dict])
async def delete_dataset(
    robot_id: str, dataset_id: str,
    db: AsyncSession = Depends(get_db), _=Depends(verify_robot_access),
):
    result = await db.execute(
        select(VisionDataset).where(VisionDataset.robot_id == robot_id, VisionDataset.dataset_id == dataset_id)
    )
    ds = result.scalar_one_or_none()
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    await db.delete(ds)
    await db.commit()
    return ResponseModel(success=True, data={"deleted": True})
