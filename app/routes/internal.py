from fastapi import APIRouter, Depends, HTTPException
from app.middleware.internal_auth import verify_internal_token
from app.services.vision_service import vision_service
from app.schemas.common import ResponseModel

router = APIRouter(tags=["Internal"])


@router.get("/context", response_model=ResponseModel[dict])
async def get_vision_context(
    robot_id: str,
    _=Depends(verify_internal_token),
):
    """Return visual context for the Voice Assistant and Agent Runtime.
    
    Requires X-Internal-Service-Token header.
    """
    state = await vision_service.get_latest_state(robot_id)
    if not state:
        raise HTTPException(status_code=404, detail="No vision state available")

    scene = state.get("scene", {})
    context = {
        "robot_id": robot_id,
        "timestamp": state.get("timestamp"),
        "scene_description": state.get("description", ""),
        "visible_people": [
            {"name": f.get("display_name", "Unknown"), "status": f.get("status")}
            for f in state.get("faces", [])
        ],
        "objects": scene.get("important_objects", []),
        "gestures": [g.get("gesture_name") for g in state.get("gestures", [])],
        "safety_observations": [scene.get("security_observation")]
        if scene.get("security_observation") else [],
        "low_light": state.get("low_light", False),
        "performance": state.get("performance", {}),
    }
    return ResponseModel(success=True, data=context)
