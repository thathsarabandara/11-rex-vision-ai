import asyncio
import logging
import time
from typing import Optional
import numpy as np
from app.config.settings import settings

logger = logging.getLogger(__name__)

# Shared queue that inference_worker reads from
frame_queue: asyncio.Queue = asyncio.Queue(maxsize=10)


class CameraWorker:
    """Opens a camera source and feeds frames into the shared frame queue.

    Supports USB_CAMERA and MJPEG_STREAM. Drops frames if the queue is full
    to avoid memory accumulation.
    """

    def __init__(self) -> None:
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self, source_type: str = "USB_CAMERA", source_url: Optional[str] = None) -> None:
        self._running = True
        self._task = asyncio.create_task(self._capture_loop(source_type, source_url))
        logger.info(f"CameraWorker started: source_type={source_type}")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("CameraWorker stopped")

    async def _capture_loop(self, source_type: str, source_url: Optional[str]) -> None:
        loop = asyncio.get_event_loop()
        cap = None
        try:
            import cv2  # type: ignore
            source = 0 if source_type == "USB_CAMERA" else (source_url or 0)
            cap = cv2.VideoCapture(source)

            if not cap.isOpened():
                logger.error(f"Failed to open camera source: {source}")
                return

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.VISION_INPUT_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.VISION_INPUT_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, settings.VISION_MAX_FPS)

            frame_interval = 1.0 / settings.VISION_MAX_FPS

            while self._running:
                t0 = time.monotonic()
                ret, frame = await loop.run_in_executor(None, cap.read)
                if not ret:
                    logger.warning("Frame read failed — retrying")
                    await asyncio.sleep(0.5)
                    continue

                if not frame_queue.full():
                    await frame_queue.put(frame)
                else:
                    # Drop frame to prevent queue backup
                    pass

                elapsed = time.monotonic() - t0
                sleep = max(0.0, frame_interval - elapsed)
                await asyncio.sleep(sleep)

        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error(f"Camera capture error: {exc}")
        finally:
            if cap is not None:
                import cv2  # type: ignore
                cap.release()


camera_worker = CameraWorker()
