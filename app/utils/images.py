import io
import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MIN_FACE_SIZE = 48


def validate_image_bytes(data: bytes, content_type: str) -> Optional[str]:
    """Return error string or None if valid."""
    if content_type not in ALLOWED_MIME_TYPES:
        return f"Unsupported image type: {content_type}"
    if len(data) > MAX_IMAGE_BYTES:
        return f"Image too large: {len(data)} bytes (max {MAX_IMAGE_BYTES})"
    return None


def decode_image(data: bytes) -> Optional[np.ndarray]:
    """Decode image bytes to numpy BGR array."""
    try:
        import cv2  # type: ignore
        arr = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return frame
    except Exception as exc:
        logger.error(f"Image decode error: {exc}")
        return None


def check_blur(frame: np.ndarray) -> float:
    """Return Laplacian variance (higher = sharper)."""
    try:
        import cv2  # type: ignore
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception:
        return 0.0


def is_too_blurry(frame: np.ndarray, threshold: float = 50.0) -> bool:
    return check_blur(frame) < threshold
