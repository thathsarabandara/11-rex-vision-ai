import pytest


@pytest.mark.asyncio
async def test_health_live(client):
    response = await client.get("/health/live")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["service"] == "rex-vision-ai"


@pytest.mark.asyncio
async def test_health_ready(client):
    response = await client.get("/health/ready")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "checks" in data
    assert "models" in data["checks"]


@pytest.mark.asyncio
async def test_health_gpu(client):
    response = await client.get("/health/gpu")
    assert response.status_code == 200
    data = response.json()
    assert "gpu_available" in data
