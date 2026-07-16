import pytest


ROBOT_ID = "robot-api-001"
VISION_PREFIX = f"/api/v1/robots/{ROBOT_ID}/vision"


@pytest.mark.asyncio
async def test_get_features_unauthorized(client):
    response = await client.get(f"{VISION_PREFIX}/features")
    assert response.status_code in (401, 403)


@pytest.mark.asyncio
async def test_get_features_robot_not_owned(client, auth_headers, mock_ownership_denied):
    response = await client.get(f"{VISION_PREFIX}/features", headers=auth_headers)
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_get_features_success(client, auth_headers, mock_ownership_ok):
    response = await client.get(f"{VISION_PREFIX}/features", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "object_detection" in data["data"]


@pytest.mark.asyncio
async def test_update_features_invalid_dependency(client, auth_headers, mock_ownership_ok):
    response = await client.put(
        f"{VISION_PREFIX}/features",
        json={"face_recognition": True, "face_detection": False},
        headers=auth_headers,
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_update_features_valid(client, auth_headers, mock_ownership_ok):
    response = await client.put(
        f"{VISION_PREFIX}/features",
        json={"scene_description": False},
        headers=auth_headers,
    )
    assert response.status_code == 200
    assert response.json()["success"] is True


@pytest.mark.asyncio
async def test_get_latest_state_no_state(client, auth_headers, mock_ownership_ok):
    response = await client.get(f"{VISION_PREFIX}/latest", headers=auth_headers)
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_get_objects_empty(client, auth_headers, mock_ownership_ok):
    response = await client.get(f"{VISION_PREFIX}/objects", headers=auth_headers)
    assert response.status_code == 200
    assert response.json()["data"] == []


@pytest.mark.asyncio
async def test_get_people_empty(client, auth_headers, mock_ownership_ok):
    response = await client.get(f"{VISION_PREFIX}/people", headers=auth_headers)
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_get_scene_empty(client, auth_headers, mock_ownership_ok):
    response = await client.get(f"{VISION_PREFIX}/scene", headers=auth_headers)
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_get_performance_empty(client, auth_headers, mock_ownership_ok):
    response = await client.get(f"{VISION_PREFIX}/performance", headers=auth_headers)
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_create_camera_source(client, auth_headers, mock_ownership_ok):
    response = await client.post(
        f"{VISION_PREFIX}/cameras",
        json={"source_type": "USB_CAMERA", "enabled": True, "resolution": "1280x720", "target_fps": 30},
        headers=auth_headers,
    )
    assert response.status_code == 200
    assert response.json()["data"]["source_type"] == "USB_CAMERA"


@pytest.mark.asyncio
async def test_list_camera_sources(client, auth_headers, mock_ownership_ok):
    response = await client.get(f"{VISION_PREFIX}/cameras", headers=auth_headers)
    assert response.status_code == 200
    assert isinstance(response.json()["data"], list)


@pytest.mark.asyncio
async def test_get_events_empty(client, auth_headers, mock_ownership_ok):
    response = await client.get(f"{VISION_PREFIX}/events", headers=auth_headers)
    assert response.status_code == 200
    assert response.json()["data"] == []
