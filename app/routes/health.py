from fastapi import APIRouter
from app.config.database import engine
from app.config.redis import get_redis
from app.ai.model_manager import model_manager

router = APIRouter(tags=["Health"])


@router.get("/health/live")
async def liveness():
    return {"status": "ok", "service": "rex-vision-ai"}


@router.get("/health/ready")
async def readiness():
    checks: dict = {}

    # DB check
    try:
        async with engine.connect() as conn:
            await conn.execute(__import__("sqlalchemy").text("SELECT 1"))
        checks["database"] = "ok"
    except Exception:
        checks["database"] = "unavailable"

    # Redis check
    try:
        redis = await get_redis()
        await redis.ping()
        checks["redis"] = "ok"
    except Exception:
        checks["redis"] = "unavailable"

    checks["models"] = {
        s["model_key"]: s["status"]
        for s in model_manager.get_status()
    }

    ready = checks["database"] == "ok" and checks["redis"] == "ok"
    return {"status": "ready" if ready else "degraded", "checks": checks}


@router.get("/health/gpu")
async def gpu_status():
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            devices = []
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                devices.append({
                    "index": i,
                    "name": props.name,
                    "total_memory_mb": round(props.total_memory / 1e6, 1),
                    "allocated_memory_mb": round(torch.cuda.memory_allocated(i) / 1e6, 1),
                })
            return {"gpu_available": True, "devices": devices}
        return {"gpu_available": False, "reason": "CUDA not available"}
    except ImportError:
        return {"gpu_available": False, "reason": "PyTorch not installed"}
