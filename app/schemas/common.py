from pydantic import BaseModel, Field
from typing import Optional, Generic, TypeVar

T = TypeVar("T")


class ResponseModel(BaseModel, Generic[T]):
    success: bool
    data: Optional[T] = None
    error: Optional[dict] = None


class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: Optional[str] = None


class PaginatedResponse(BaseModel, Generic[T]):
    success: bool
    data: list[T]
    total: int
    limit: int
    offset: int
