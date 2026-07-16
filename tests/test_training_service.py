import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_create_training_job_success(mock_ownership_ok):
    from app.services.training_service import TrainingService
    from tests.conftest import TestSessionLocal
    service = TrainingService()
    with patch.object(service, "_run_job", new=AsyncMock()):
        async with TestSessionLocal() as db:
            job = await service.create_job(
                robot_id="robot-train-001", user_id="user-001",
                training_type="OBJECT_DETECTION",
                dataset_id=None, base_model=None,
                epochs=10, image_size=640, db=db,
            )
            assert job.status == "QUEUED"
            assert job.training_type == "OBJECT_DETECTION"
            assert job.epochs == 10


@pytest.mark.asyncio
async def test_create_training_job_locks_concurrent_jobs(mock_ownership_ok):
    from app.services.training_service import TrainingService
    from tests.conftest import TestSessionLocal
    service = TrainingService()

    with patch.object(service, "_run_job", new=AsyncMock()):
        async with TestSessionLocal() as db:
            await service.create_job(
                robot_id="robot-train-lock", user_id="user-001",
                training_type="GESTURE_CLASSIFICATION",
                dataset_id=None, base_model=None,
                epochs=5, image_size=640, db=db,
            )

        # Simulate Redis lock as if job is running
        from app.config.redis import get_redis
        redis = await get_redis()
        await redis.set("rex:vision:training-lock:robot-train-lock", "existing-job", ex=3600)

        async with TestSessionLocal() as db:
            with pytest.raises(ValueError, match="training job is already running"):
                await service.create_job(
                    robot_id="robot-train-lock", user_id="user-001",
                    training_type="OBJECT_DETECTION",
                    dataset_id=None, base_model=None,
                    epochs=5, image_size=640, db=db,
                )


@pytest.mark.asyncio
async def test_list_training_jobs_empty(mock_ownership_ok):
    from app.services.training_service import TrainingService
    from tests.conftest import TestSessionLocal
    service = TrainingService()
    async with TestSessionLocal() as db:
        jobs = await service.list_jobs("robot-no-jobs", db)
        assert jobs == []


@pytest.mark.asyncio
async def test_get_nonexistent_job_returns_none():
    from app.services.training_service import TrainingService
    from tests.conftest import TestSessionLocal
    service = TrainingService()
    async with TestSessionLocal() as db:
        job = await service.get_job("robot-001", "no-such-job", db)
        assert job is None


@pytest.mark.asyncio
async def test_cancel_nonexistent_job_returns_false():
    from app.services.training_service import TrainingService
    from tests.conftest import TestSessionLocal
    service = TrainingService()
    async with TestSessionLocal() as db:
        result = await service.cancel_job("robot-001", "no-such-job", db)
        assert result is False


@pytest.mark.asyncio
async def test_cancel_queued_job_succeeds():
    from app.services.training_service import TrainingService
    from tests.conftest import TestSessionLocal
    service = TrainingService()
    with patch.object(service, "_run_job", new=AsyncMock()):
        async with TestSessionLocal() as db:
            job = await service.create_job(
                robot_id="robot-cancel-001", user_id="user-001",
                training_type="SCENE_CLASSIFIER",
                dataset_id=None, base_model=None,
                epochs=5, image_size=640, db=db,
            )
            result = await service.cancel_job("robot-cancel-001", job.job_id, db)
            assert result is True

            cancelled = await service.get_job("robot-cancel-001", job.job_id, db)
            assert cancelled.status == "CANCELLED"
