from pydantic import BaseModel, Field
from typing import Optional


class MemorySearchIn(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    limit: int = Field(default=10, ge=1, le=50)


class MemorySearchResult(BaseModel):
    scene_id: str
    description: str
    captured_at: str
    similarity: float
    media_url: Optional[str] = None
