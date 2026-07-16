import pytest


ROBOT_ID = "robot-face-route-001"
FACE_PREFIX = f"/api/v1/robots/{ROBOT_ID}/vision/faces"


@pytest.mark.asyncio
async def test_list_faces_unauthorized(client):
    response = await client.get(FACE_PREFIX)
    assert response.status_code in (401, 403)


@pytest.mark.asyncio
async def test_list_faces_robot_not_owned(client, auth_headers, mock_ownership_denied):
    response = await client.get(FACE_PREFIX, headers=auth_headers)
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_list_faces_empty(client, auth_headers, mock_ownership_ok):
    response = await client.get(FACE_PREFIX, headers=auth_headers)
    assert response.status_code == 200
    assert response.json()["data"] == []


@pytest.mark.asyncio
async def test_register_face_no_images(client, auth_headers, mock_ownership_ok, mock_qdrant, mock_minio):
    response = await client.post(
        FACE_PREFIX,
        data={"display_name": "Test", "relationship": "OWNER"},
        headers=auth_headers,
    )
    # Should fail due to missing images
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_get_nonexistent_face(client, auth_headers, mock_ownership_ok):
    response = await client.get(f"{FACE_PREFIX}/no-such-id", headers=auth_headers)
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_delete_nonexistent_face(client, auth_headers, mock_ownership_ok, mock_qdrant, mock_minio):
    response = await client.delete(f"{FACE_PREFIX}/no-such-id", headers=auth_headers)
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_patch_nonexistent_face(client, auth_headers, mock_ownership_ok):
    response = await client.patch(
        f"{FACE_PREFIX}/no-such-id",
        json={"display_name": "New Name"},
        headers=auth_headers,
    )
    assert response.status_code == 404
