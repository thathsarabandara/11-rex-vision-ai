import asyncio
import json
import logging
from fastapi import WebSocket, WebSocketDisconnect
from app.config.settings import settings
import jwt

logger = logging.getLogger(__name__)


class WebSocketService:
    """Registry and broadcaster for per-robot WebSocket connections."""

    def __init__(self) -> None:
        # robot_id → list of connected websocket connections
        self._connections: dict[str, list[WebSocket]] = {}

    def authenticate(self, token: str) -> dict | None:
        """Validate JWT and return payload or None."""
        try:
            payload = jwt.decode(
                token,
                settings.USER_JWT_SECRET_KEY,
                algorithms=[settings.USER_JWT_ALGORITHM],
                issuer=settings.USER_JWT_ISSUER,
                audience=settings.USER_JWT_AUDIENCE,
            )
            if not payload.get("email_verified"):
                return None
            return payload
        except jwt.InvalidTokenError:
            return None

    async def connect(self, robot_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        self._connections.setdefault(robot_id, []).append(websocket)
        logger.info(f"WS client connected: robot={robot_id} total={len(self._connections[robot_id])}")

    def disconnect(self, robot_id: str, websocket: WebSocket) -> None:
        conns = self._connections.get(robot_id, [])
        if websocket in conns:
            conns.remove(websocket)
        logger.info(f"WS client disconnected: robot={robot_id}")

    async def broadcast(self, robot_id: str, message: dict) -> None:
        """Broadcast a JSON message to all connected clients for a robot.
        
        Disconnected clients are silently removed to avoid blocking.
        """
        conns = self._connections.get(robot_id, [])
        if not conns:
            return

        dead: list[WebSocket] = []
        text = json.dumps(message)
        for ws in conns:
            try:
                await asyncio.wait_for(ws.send_text(text), timeout=1.0)
            except Exception:
                dead.append(ws)

        for ws in dead:
            self.disconnect(robot_id, ws)

    def connection_count(self, robot_id: str) -> int:
        return len(self._connections.get(robot_id, []))


websocket_service = WebSocketService()
