import pytest


ROBOT_ID = "robot-training-route-001"
TRAINING_PREFIX = f"/api/v1/robots/{ROBOT_ID}/vision/training/jobs"


@pytest.mark.asyncio
async def test_create_training_job_unauthorized(client):
    response = await client.post(TRAINING_PREFIX, json={"training_type": "OBJECT_DETECTION"})
    assert response.status_code in (401, 403)


@pytest.mark.asyncio
async def test_create_training_job_robot_not_owned(client, auth_headers, mock_ownership_denied):
    response = await client.post(TRAINING_PREFIX, json={"training_type": "OBJECT_DETECTION"}, headers=auth_headers)
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_create_training_job_success(client, auth_headers, mock_ownership_ok):
    from unittest.mock import AsyncMock, patch
    from app.services.training_service import TrainingService
    with patch.object(TrainingService, "_run_job", new=AsyncMock()):
        response = await client.post(
            TRAINING_PREFIX,
            json={"training_type": "GESTURE_CLASSIFICATION", "epochs": 20},
            headers=auth_headers,
        )
        assert response.status_code == 202
        data = response.json()
        assert data["success"] is True
        assert data["data"]["status"] == "QUEUED"
        assert data["data"]["training_type"] == "GESTURE_CLASSIFICATION"


@pytest.mark.asyncio
async def test_list_training_jobs_empty(client, auth_headers, mock_ownership_ok):
    response = await client.get(f"/api/v1/robots/robot-no-jobs-at-all/vision/training/jobs", headers=auth_headers)
    assert response.status_code == 200
    assert response.json()["data"] == []


@pytest.mark.asyncio
async def test_get_nonexistent_training_job(client, auth_headers, mock_ownership_ok):
    response = await client.get(f"{TRAINING_PREFIX}/no-such-job", headers=auth_headers)
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_cancel_nonexistent_training_job(client, auth_headers, mock_ownership_ok):
    response = await client.post(f"{TRAINING_PREFIX}/no-such-job/cancel", headers=auth_headers)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_dataset_crud(client, auth_headers, mock_ownership_ok):
    DATASET_PREFIX = f"/api/v1/robots/{ROBOT_ID}/vision/datasets"
    # Create
    response = await client.post(
        DATASET_PREFIX,
        json={"name": "Test Dataset", "dataset_type": "OBJECT_DETECTION"},
        headers=auth_headers,
    )
    assert response.status_code == 201
    dataset_id = response.json()["data"]["dataset_id"]

    # Get
    response = await client.get(f"{DATASET_PREFIX}/{dataset_id}", headers=auth_headers)
    assert response.status_code == 200
    assert response.json()["data"]["name"] == "Test Dataset"

    # List
    response = await client.get(DATASET_PREFIX, headers=auth_headers)
    assert response.status_code == 200

    # Delete
    response = await client.delete(f"{DATASET_PREFIX}/{dataset_id}", headers=auth_headers)
    assert response.status_code == 200
    assert response.json()["data"]["deleted"] is True
