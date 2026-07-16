import pytest


ROBOT_ID = "robot-gesture-route-001"
GESTURE_PREFIX = f"/api/v1/robots/{ROBOT_ID}/vision/gestures"


@pytest.mark.asyncio
async def test_list_gestures_unauthorized(client):
    response = await client.get(GESTURE_PREFIX)
    assert response.status_code in (401, 403)


@pytest.mark.asyncio
async def test_list_gestures_robot_not_owned(client, auth_headers, mock_ownership_denied):
    response = await client.get(GESTURE_PREFIX, headers=auth_headers)
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_list_gestures_empty(client, auth_headers, mock_ownership_ok):
    response = await client.get(GESTURE_PREFIX, headers=auth_headers)
    assert response.status_code == 200
    assert response.json()["data"] == []


@pytest.mark.asyncio
async def test_register_gesture_no_samples(client, auth_headers, mock_ownership_ok, mock_qdrant, mock_minio):
    response = await client.post(
        GESTURE_PREFIX,
        data={"name": "THUMBS_UP"},
        headers=auth_headers,
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_get_nonexistent_gesture(client, auth_headers, mock_ownership_ok):
    response = await client.get(f"{GESTURE_PREFIX}/no-such-id", headers=auth_headers)
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_delete_nonexistent_gesture(client, auth_headers, mock_ownership_ok, mock_qdrant, mock_minio):
    response = await client.delete(f"{GESTURE_PREFIX}/no-such-id", headers=auth_headers)
    assert response.status_code == 404
