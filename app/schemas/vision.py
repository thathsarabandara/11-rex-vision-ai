from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class BoundingBox(BaseModel):
    x: int
    y: int
    width: int
    height: int


class Detection(BaseModel):
    track_id: Optional[int] = None
    class_id: int
    class_name: str
    confidence: float
    bounding_box: BoundingBox


class PerformanceMetrics(BaseModel):
    fps: float
    pipeline_ms: float
    gpu_memory_mb: Optional[float] = None


class VisionState(BaseModel):
    frame_id: str
    timestamp: str
    robot_id: str
    detections: list[Detection] = []
    faces: list[dict] = []
    expressions: list[dict] = []
    gestures: list[dict] = []
    movements: list[dict] = []
    scene: dict = {}
    performance: PerformanceMetrics


class FeatureFlagsIn(BaseModel):
    object_detection: Optional[bool] = None
    face_detection: Optional[bool] = None
    face_recognition: Optional[bool] = None
    expression_estimation: Optional[bool] = None
    person_tracking: Optional[bool] = None
    gesture_detection: Optional[bool] = None
    scene_understanding: Optional[bool] = None
    scene_description: Optional[bool] = None
    visual_obstacle_detection: Optional[bool] = None
    low_light_detection: Optional[bool] = None
    incident_capture: Optional[bool] = None


class FeatureFlagsOut(BaseModel):
    object_detection: bool
    face_detection: bool
    face_recognition: bool
    expression_estimation: bool
    person_tracking: bool
    gesture_detection: bool
    scene_understanding: bool
    scene_description: bool
    visual_obstacle_detection: bool
    low_light_detection: bool
    incident_capture: bool


class CameraSourceIn(BaseModel):
    source_type: str
    source_url: Optional[str] = None
    enabled: bool = True
    resolution: str = "640x480"
    target_fps: int = Field(default=20, ge=1, le=60)


class CameraSourceOut(BaseModel):
    source_id: str
    robot_id: str
    source_type: str
    source_url: Optional[str]
    enabled: bool
    resolution: str
    target_fps: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class VisionEventOut(BaseModel):
    event_id: str
    robot_id: str
    event_type: str
    severity: str
    data: dict
    occurred_at: datetime

    class Config:
        from_attributes = True
