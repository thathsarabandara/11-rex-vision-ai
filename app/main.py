import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app
import logging

from app.config.settings import settings
from app.config.logging import setup_logging
from app.config.database import engine
from app.config.storage import ensure_bucket
from app.middleware.request_id import RequestIdMiddleware
from app.middleware.error_handler import ErrorHandlerMiddleware
from app.middleware.rate_limit import RateLimitMiddleware
from app.services.kafka_service import kafka_service
from app.ai.model_manager import model_manager
from app.workers.cleanup_worker import start_cleanup_worker, stop_cleanup_worker

from app.routes import vision, faces, gestures, tracking, models, datasets, training, memory, websockets, internal, health

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    logger.info(f"Starting {settings.APP_NAME} in {settings.APP_ENV} mode")

    # Kafka producer
    await kafka_service.start()

    # Storage bucket
    try:
        ensure_bucket()
    except Exception as exc:
        logger.warning(f"MinIO bucket init failed (non-fatal): {exc}")

    # Load enabled AI models
    try:
        model_manager.load_all_enabled()
    except Exception as exc:
        logger.warning(f"Model loading error (non-fatal): {exc}")

    # Cleanup worker
    await start_cleanup_worker()

    yield

    logger.info("Shutting down rex-vision-ai...")
    await kafka_service.stop()
    await stop_cleanup_worker()


app = FastAPI(
    title="REX Vision AI Service",
    description="Visual-perception engine for the REX Smart Home Robot Platform",
    version="1.0.0",
    lifespan=lifespan,
)

# Middleware
app.add_middleware(RequestIdMiddleware)
app.add_middleware(ErrorHandlerMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# API routes
PREFIX = "/api/v1/robots/{robot_id}/vision"
app.include_router(vision.router,    prefix=PREFIX)
app.include_router(faces.router,     prefix=f"{PREFIX}/faces")
app.include_router(gestures.router,  prefix=f"{PREFIX}/gestures")
app.include_router(tracking.router,  prefix=f"{PREFIX}/tracking")
app.include_router(models.router,    prefix=f"{PREFIX}/models")
app.include_router(datasets.router,  prefix=f"{PREFIX}/datasets")
app.include_router(training.router,  prefix=f"{PREFIX}/training/jobs")
app.include_router(memory.router,    prefix=f"{PREFIX}/memory")

# WebSocket
app.include_router(websockets.router, prefix="/api/v1/ws/robots/{robot_id}/vision")

# Internal
app.include_router(internal.router, prefix="/internal/v1/robots/{robot_id}/vision")

# Health
app.include_router(health.router)
