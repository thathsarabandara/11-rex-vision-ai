import asyncio
import logging
from app.models.training_job import TrainingJob

logger = logging.getLogger(__name__)


async def execute_training(job: TrainingJob) -> dict:
    """Execute a training job in the background.

    Returns a metrics dictionary. Training runs in a separate thread via
    run_in_executor to avoid blocking the event loop.
    """
    loop = asyncio.get_event_loop()
    metrics = await loop.run_in_executor(None, _sync_train, job)
    return metrics


def _sync_train(job: TrainingJob) -> dict:
    """Synchronous training execution. Runs in a thread pool."""
    logger.info(f"Training job {job.job_id} starting — type={job.training_type}")

    if job.training_type == "OBJECT_DETECTION":
        return _train_object_detection(job)
    elif job.training_type == "GESTURE_CLASSIFICATION":
        return _train_gesture(job)
    else:
        logger.warning(f"No training handler for type: {job.training_type}")
        return {"message": f"No trainer configured for {job.training_type}"}


def _train_object_detection(job: TrainingJob) -> dict:
    """Placeholder for YOLO fine-tuning via Ultralytics."""
    try:
        from ultralytics import YOLO  # type: ignore
        from app.config.settings import settings

        base = job.base_model or settings.OBJECT_MODEL_PATH
        model = YOLO(base)
        # Real dataset path would be fetched from MinIO
        # model.train(data="...", epochs=job.epochs, imgsz=job.image_size)
        logger.info(f"Object detection training completed (stub): job={job.job_id}")
        return {
            "mAP50": 0.0,
            "mAP50_95": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "epochs": job.epochs,
            "note": "Stub result — attach real dataset to enable full training",
        }
    except Exception as exc:
        raise RuntimeError(f"Object detection training failed: {exc}") from exc


def _train_gesture(job: TrainingJob) -> dict:
    """Placeholder for gesture classifier training."""
    return {
        "accuracy": 0.0,
        "note": "Stub result — attach gesture dataset to enable full training",
    }
