from fastapi import Request, HTTPException, status
from app.config.settings import settings


async def verify_internal_token(request: Request) -> None:
    """Dependency that validates the X-Internal-Service-Token header."""
    token = request.headers.get("X-Internal-Service-Token")
    if not token or token != settings.INTERNAL_SERVICE_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid internal service token",
        )
