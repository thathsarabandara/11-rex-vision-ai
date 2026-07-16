from pydantic import BaseModel
from typing import Optional
from app.schemas.vision import BoundingBox


class TrackingTargetIn(BaseModel):
    track_id: int


class TrackingTargetOut(BaseModel):
    track_id: Optional[int]
    robot_id: str
    active: bool


class TrackedPerson(BaseModel):
    track_id: int
    class_name: str
    confidence: float
    bounding_box: BoundingBox
    age_frames: int
    lost_frames: int
    center_x: Optional[float] = None
    center_y: Optional[float] = None


class MovementResult(BaseModel):
    track_id: int
    movement: str
    velocity_x: float
    velocity_y: float
    scale_change: float
    confidence: float
