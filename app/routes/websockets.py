import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from app.services.websocket_service import websocket_service
from app.services.vision_service import vision_service

logger = logging.getLogger(__name__)
router = APIRouter(tags=["WebSocket"])


@router.websocket("")
async def vision_websocket(robot_id: str, websocket: WebSocket, token: str = Query(...)):
    """Live vision metadata stream for a robot.

    Authentication via `token` query parameter containing a user JWT.
    Ownership is validated before connection is accepted.
    """
    payload = websocket_service.authenticate(token)
    if not payload:
        await websocket.close(code=4001, reason="Unauthorized")
        return

    # Verify robot ownership
    from app.services.ownership_service import ownership_service
    user_id = payload.get("sub")
    owned = await ownership_service.verify_ownership(user_id, robot_id)
    if not owned:
        await websocket.close(code=4003, reason="Robot not owned")
        return

    await websocket_service.connect(robot_id, websocket)

    # Send latest state immediately on connect
    from app.config.database import AsyncSessionLocal
    async with AsyncSessionLocal() as db:
        state = await vision_service.get_latest_state(robot_id)
    if state:
        try:
            await websocket.send_json({"type": "vision.state.updated", "data": state})
        except Exception:
            pass

    try:
        while True:
            # Keep the connection alive; inference worker broadcasts updates
            await websocket.receive_text()
    except WebSocketDisconnect:
        websocket_service.disconnect(robot_id, websocket)
    except Exception as exc:
        logger.warning(f"WebSocket error: {exc}")
        websocket_service.disconnect(robot_id, websocket)
