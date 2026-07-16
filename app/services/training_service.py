import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.config.redis import get_redis
from app.models.training_job import TrainingJob
from app.services.kafka_service import kafka_service
from app.config.kafka import TOPIC_TRAINING_COMPLETED, TOPIC_TRAINING_FAILED
from app.config.settings import settings

logger = logging.getLogger(__name__)


class TrainingService:
    async def create_job(
        self,
        robot_id: str,
        user_id: str,
        training_type: str,
        dataset_id: Optional[str],
        base_model: Optional[str],
        epochs: Optional[int],
        image_size: int,
        db: AsyncSession,
    ) -> TrainingJob:
        # Enforce concurrency limit via Redis lock
        redis = await get_redis()
        lock_key = f"rex:vision:training-lock:{robot_id}"
        existing = await redis.get(lock_key)
        if existing:
            raise ValueError("A training job is already running for this robot")

        job = TrainingJob(
            job_id=str(uuid.uuid4()),
            robot_id=robot_id,
            user_id=user_id,
            training_type=training_type,
            dataset_id=dataset_id,
            base_model=base_model,
            epochs=epochs or 30,
            image_size=image_size,
            status="QUEUED",
        )
        db.add(job)
        await db.commit()
        await db.refresh(job)

        # Dispatch background worker
        asyncio.create_task(self._run_job(job.job_id, robot_id))
        return job

    async def _run_job(self, job_id: str, robot_id: str) -> None:
        """Background training task. Updates status in DB and publishes events."""
        from app.config.database import AsyncSessionLocal
        redis = await get_redis()
        lock_key = f"rex:vision:training-lock:{robot_id}"

        async with AsyncSessionLocal() as db:
            result = await db.execute(select(TrainingJob).where(TrainingJob.job_id == job_id))
            job = result.scalar_one_or_none()
            if not job:
                return

            await redis.set(lock_key, job_id, ex=3600)
            job.status = "RUNNING"
            job.started_at = datetime.now(timezone.utc)
            await db.commit()

            try:
                # Delegate to background training worker
                from app.workers.training_worker import execute_training
                metrics = await execute_training(job)
                job.status = "COMPLETED"
                job.metrics = metrics
                job.completed_at = datetime.now(timezone.utc)
                await db.commit()
                await kafka_service.publish(TOPIC_TRAINING_COMPLETED, {
                    "job_id": job_id,
                    "robot_id": robot_id,
                    "training_type": job.training_type,
                    "metrics": metrics,
                })
            except Exception as exc:
                logger.error(f"Training job {job_id} failed: {exc}")
                job.status = "FAILED"
                job.logs = str(exc)
                job.completed_at = datetime.now(timezone.utc)
                await db.commit()
                await kafka_service.publish(TOPIC_TRAINING_FAILED, {
                    "job_id": job_id,
                    "robot_id": robot_id,
                    "error": str(exc),
                })
            finally:
                await redis.delete(lock_key)

    async def list_jobs(self, robot_id: str, db: AsyncSession) -> list[TrainingJob]:
        result = await db.execute(
            select(TrainingJob).where(TrainingJob.robot_id == robot_id)
            .order_by(TrainingJob.created_at.desc())
        )
        return list(result.scalars().all())

    async def get_job(self, robot_id: str, job_id: str, db: AsyncSession) -> Optional[TrainingJob]:
        result = await db.execute(
            select(TrainingJob).where(
                TrainingJob.robot_id == robot_id,
                TrainingJob.job_id == job_id,
            )
        )
        return result.scalar_one_or_none()

    async def cancel_job(self, robot_id: str, job_id: str, db: AsyncSession) -> bool:
        job = await self.get_job(robot_id, job_id, db)
        if not job or job.status not in ("QUEUED", "RUNNING"):
            return False
        job.status = "CANCELLED"
        job.completed_at = datetime.now(timezone.utc)
        await db.commit()
        redis = await get_redis()
        await redis.delete(f"rex:vision:training-lock:{robot_id}")
        return True


training_service = TrainingService()
