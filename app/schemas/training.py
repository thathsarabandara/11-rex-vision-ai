from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class TrainingJobCreateIn(BaseModel):
    training_type: str  # OBJECT_DETECTION, GESTURE_CLASSIFICATION, FACE_PROFILE_UPDATE, SCENE_CLASSIFIER
    dataset_id: Optional[str] = None
    base_model: Optional[str] = None
    epochs: Optional[int] = Field(None, ge=1, le=500)
    image_size: Optional[int] = Field(default=640)


class TrainingJobOut(BaseModel):
    job_id: str
    robot_id: str
    training_type: str
    dataset_id: Optional[str]
    base_model: Optional[str]
    epochs: Optional[int]
    image_size: Optional[int]
    status: str
    logs: Optional[str]
    metrics: Optional[dict]
    artifact_path: Optional[str]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    created_at: datetime

    class Config:
        from_attributes = True
