from fastapi.responses import JSONResponse
from app.schemas.common import ResponseModel


def ok(data=None) -> JSONResponse:
    return JSONResponse(content={"success": True, "data": data})


def error(code: str, message: str, status_code: int = 400, request_id: str | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "error": {"code": code, "message": message, "request_id": request_id},
        },
    )
