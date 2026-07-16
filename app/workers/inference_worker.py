import asyncio
import logging
import time
from collections import deque
from typing import Optional
from app.workers.camera_worker import frame_queue
from app.pipelines.frame_pipeline import run_frame_pipeline
from app.services.vision_service import vision_service
from app.services.websocket_service import websocket_service
from app.services.event_service import event_service
from app.services.kafka_service import kafka_service
from app.config.kafka import (
    TOPIC_STATE_UPDATED,
    TOPIC_FACE_RECOGNIZED,
    TOPIC_UNKNOWN_PERSON,
    TOPIC_GESTURE_DETECTED,
    TOPIC_FALL_CANDIDATE,
    TOPIC_LOW_LIGHT,
    TOPIC_CONTEXT_UPDATED,
)

logger = logging.getLogger(__name__)

# Rolling FPS calculation
FPS_WINDOW = 30


class InferenceWorker:
    def __init__(self) -> None:
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._fps_timestamps: deque = deque(maxlen=FPS_WINDOW)

    async def start(self, robot_id: str) -> None:
        self._running = True
        self._task = asyncio.create_task(self._inference_loop(robot_id))
        logger.info(f"InferenceWorker started: robot={robot_id}")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _inference_loop(self, robot_id: str) -> None:
        from app.config.database import AsyncSessionLocal

        while self._running:
            try:
                frame = await asyncio.wait_for(frame_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            try:
                async with AsyncSessionLocal() as db:
                    features = await vision_service.get_features(robot_id, db)
                    face_profiles: list[dict] = []
                    gesture_profiles: list[dict] = []

                    if features.get("face_detection"):
                        from app.services.face_service import face_service
                        face_profiles = await face_service.get_active_embeddings(robot_id)

                    if features.get("gesture_detection"):
                        from app.services.gesture_service import gesture_service
                        gesture_profiles = await gesture_service.get_active_profiles(robot_id)

                state = await run_frame_pipeline(
                    frame=frame,
                    robot_id=robot_id,
                    features=features,
                    face_profiles=face_profiles,
                    gesture_profiles=gesture_profiles,
                )

                # Update FPS
                now = time.monotonic()
                self._fps_timestamps.append(now)
                if len(self._fps_timestamps) >= 2:
                    elapsed = self._fps_timestamps[-1] - self._fps_timestamps[0]
                    fps = len(self._fps_timestamps) / elapsed if elapsed > 0 else 0
                    state["performance"]["fps"] = round(fps, 1)

                # Store in Redis
                await vision_service.set_latest_state(robot_id, state)

                # Broadcast to WebSocket clients
                await websocket_service.broadcast(robot_id, {
                    "type": "vision.state.updated",
                    "data": state,
                })

                # Publish full state to Kafka
                await event_service.publish_always(TOPIC_STATE_UPDATED, state)

                # Publish specific events with cooldown
                await self._publish_events(robot_id, state)

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"Inference loop error: {exc}")

    async def _publish_events(self, robot_id: str, state: dict) -> None:
        # Face recognition events
        for face in state.get("faces", []):
            if face.get("status") == "KNOWN":
                await event_service.publish_if_not_cooldown(
                    robot_id, f"face.recognized.{face.get('face_profile_id')}",
                    TOPIC_FACE_RECOGNIZED, {"face": face}
                )
            elif face.get("status") == "UNKNOWN":
                await event_service.publish_if_not_cooldown(
                    robot_id, "unknown_person", TOPIC_UNKNOWN_PERSON, {"face": face}
                )

        # Gesture events
        for gesture in state.get("gestures", []):
            if gesture.get("stability_frames", 0) >= 5:
                await event_service.publish_if_not_cooldown(
                    robot_id, f"gesture.{gesture.get('gesture_name')}",
                    TOPIC_GESTURE_DETECTED, {"gesture": gesture}
                )

        # Fall candidate
        for mv in state.get("movements", []):
            if mv.get("movement") == "FALL_CANDIDATE":
                await event_service.publish_if_not_cooldown(
                    robot_id, f"fall_candidate.{mv.get('track_id')}",
                    TOPIC_FALL_CANDIDATE, {"movement": mv}, cooldown_seconds=30
                )

        # Low light
        if state.get("low_light"):
            await event_service.publish_if_not_cooldown(
                robot_id, "low_light", TOPIC_LOW_LIGHT, {}, cooldown_seconds=60
            )

        # Voice context
        scene = state.get("scene", {})
        await event_service.publish_always(TOPIC_CONTEXT_UPDATED, {
            "robot_id": robot_id,
            "timestamp": state.get("timestamp"),
            "scene_description": state.get("description", ""),
            "visible_people": [
                {"name": f.get("display_name", "Unknown"), "status": f.get("status")}
                for f in state.get("faces", [])
            ],
            "objects": scene.get("important_objects", []),
            "gestures": [g.get("gesture_name") for g in state.get("gestures", [])],
            "safety_observations": [scene.get("security_observation")] if scene.get("security_observation") else [],
        })


inference_worker = InferenceWorker()
