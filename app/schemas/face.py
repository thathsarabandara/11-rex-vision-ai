from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class FaceRegistrationIn(BaseModel):
    display_name: str = Field(..., min_length=1, max_length=100)
    relationship: str = Field(default="GUEST")


class FaceProfileOut(BaseModel):
    face_profile_id: str
    robot_id: str
    display_name: str
    relationship: str
    status: str
    sample_count: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class FaceProfilePatch(BaseModel):
    display_name: Optional[str] = Field(None, min_length=1, max_length=100)
    relationship: Optional[str] = None
    status: Optional[str] = None


class FaceRecognitionResult(BaseModel):
    face_profile_id: Optional[str] = None
    display_name: Optional[str] = None
    relationship: Optional[str] = None
    similarity: Optional[float] = None
    # KNOWN, UNKNOWN, UNCERTAIN
    status: str
