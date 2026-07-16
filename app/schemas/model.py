from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class ModelStatusOut(BaseModel):
    model_key: str
    model_version: Optional[str]
    enabled: bool
    loaded: bool
    device: str
    status: str
    last_loaded_at: Optional[datetime]
    average_inference_ms: Optional[float]


class ModelStateIn(BaseModel):
    enabled: bool
