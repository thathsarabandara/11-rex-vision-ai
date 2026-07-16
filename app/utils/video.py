import logging
from typing import Optional

logger = logging.getLogger(__name__)


def get_video_info(source: str) -> dict:
    """Return basic video source metadata without opening the stream."""
    return {"source": source, "available": True}
