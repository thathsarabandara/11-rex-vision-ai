import time
import logging
from collections import defaultdict
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

# Simple in-memory rate limiter (use Redis for production multi-instance)
_request_counts: dict[str, list[float]] = defaultdict(list)
RATE_LIMIT_REQUESTS = 200
RATE_LIMIT_WINDOW_SECONDS = 60


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        now = time.monotonic()

        bucket = _request_counts[client_ip]
        # Remove entries outside the window
        bucket[:] = [t for t in bucket if now - t < RATE_LIMIT_WINDOW_SECONDS]

        if len(bucket) >= RATE_LIMIT_REQUESTS:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
            )

        bucket.append(now)
        return await call_next(request)
