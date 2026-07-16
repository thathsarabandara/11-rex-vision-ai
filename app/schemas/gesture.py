from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class GestureRegistrationIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=60)
    description: Optional[str] = Field(None, max_length=255)
    gesture_type: str = Field(default="STATIC")  # STATIC or DYNAMIC
    action_hint: Optional[str] = Field(None, max_length=60)


class GestureProfileOut(BaseModel):
    gesture_profile_id: str
    robot_id: str
    gesture_name: str
    description: Optional[str]
    gesture_type: str
    action_hint: Optional[str]
    status: str
    is_predefined: bool
    sample_count: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class GestureProfilePatch(BaseModel):
    gesture_name: Optional[str] = Field(None, min_length=1, max_length=60)
    description: Optional[str] = Field(None, max_length=255)
    action_hint: Optional[str] = Field(None, max_length=60)
    status: Optional[str] = None


class GestureDetectionResult(BaseModel):
    gesture_profile_id: str
    gesture_name: str
    confidence: float
    handedness: str  # LEFT or RIGHT
    stability_frames: int
