from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class DatasetCreateIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    description: Optional[str] = Field(None, max_length=512)
    dataset_type: str  # OBJECT_DETECTION, GESTURE, FACE_SAMPLES, SCENE


class DatasetOut(BaseModel):
    dataset_id: str
    robot_id: str
    name: str
    description: Optional[str]
    dataset_type: str
    image_count: int
    label_count: int
    minio_prefix: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
