import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from app.config.settings import settings
from app.config.database import Base, get_db
from app.main import app

# -----------------------------------------------------------------------
# Event loop
# -----------------------------------------------------------------------
@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# -----------------------------------------------------------------------
# In-memory SQLite database for tests
# -----------------------------------------------------------------------
TEST_DB_URL = "sqlite+aiosqlite:///:memory:"
test_engine = create_async_engine(TEST_DB_URL, echo=False)
TestSessionLocal = sessionmaker(bind=test_engine, class_=AsyncSession, expire_on_commit=False)


@pytest.fixture(scope="session", autouse=True)
async def create_test_tables():
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


async def override_get_db():
    async with TestSessionLocal() as session:
        yield session


app.dependency_overrides[get_db] = override_get_db


# -----------------------------------------------------------------------
# Async test client
# -----------------------------------------------------------------------
@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


# -----------------------------------------------------------------------
# Redis mock (autouse — always active)
# -----------------------------------------------------------------------
@pytest.fixture(autouse=True)
def mock_redis():
    with patch("app.config.redis.redis_client") as mock:
        mock.get = AsyncMock(return_value=None)
        mock.set = AsyncMock()
        mock.delete = AsyncMock()
        mock.ping = AsyncMock(return_value=True)
        mock.exists = AsyncMock(return_value=0)
        yield mock


# -----------------------------------------------------------------------
# Kafka mock (autouse)
# -----------------------------------------------------------------------
@pytest.fixture(autouse=True)
def mock_kafka():
    with patch("app.services.kafka_service.kafka_service._producer") as mock:
        mock.send_and_wait = AsyncMock()
        yield mock


# -----------------------------------------------------------------------
# Ownership mock (default: owned=True)
# -----------------------------------------------------------------------
@pytest.fixture
def mock_ownership_ok():
    with patch("app.services.ownership_service.ownership_service.verify_ownership", new=AsyncMock(return_value=True)):
        yield


@pytest.fixture
def mock_ownership_denied():
    with patch("app.services.ownership_service.ownership_service.verify_ownership", new=AsyncMock(return_value=False)):
        yield


# -----------------------------------------------------------------------
# Qdrant mock
# -----------------------------------------------------------------------
@pytest.fixture
def mock_qdrant():
    with patch("app.services.vector_service.get_qdrant_client") as mock_fn:
        client = AsyncMock()
        client.upsert = AsyncMock()
        client.search = AsyncMock(return_value=[])
        client.scroll = AsyncMock(return_value=([], None))
        client.delete = AsyncMock()
        client.get_collections = AsyncMock(return_value=[])
        client.create_collection = AsyncMock()
        mock_fn.return_value = client
        yield client


# -----------------------------------------------------------------------
# MinIO mock
# -----------------------------------------------------------------------
@pytest.fixture
def mock_minio():
    with patch("app.services.storage_service.get_minio_client") as mock_fn:
        client = MagicMock()
        client.put_object = MagicMock()
        client.presigned_get_object = MagicMock(return_value="http://minio/signed-url")
        client.list_objects = MagicMock(return_value=[])
        client.remove_object = MagicMock()
        mock_fn.return_value = client
        yield client


# -----------------------------------------------------------------------
# YOLO mock
# -----------------------------------------------------------------------
@pytest.fixture
def mock_yolo():
    with patch("app.ai.object_detector.object_detector.is_loaded", True):
        with patch("app.ai.object_detector.object_detector.detect", return_value=[
            {
                "track_id": 1, "class_id": 0, "class_name": "person",
                "confidence": 0.92,
                "bounding_box": {"x": 100, "y": 50, "width": 180, "height": 400},
            }
        ]):
            yield


# -----------------------------------------------------------------------
# InsightFace mock
# -----------------------------------------------------------------------
@pytest.fixture
def mock_insightface():
    import numpy as np
    with patch("app.ai.face_detector.face_detector.detect", return_value=[
        {
            "bbox": {"x": 120, "y": 60, "width": 80, "height": 80},
            "det_score": 0.95,
            "kps": None,
            "embedding": list(np.random.randn(512).tolist()),
        }
    ]):
        yield


# -----------------------------------------------------------------------
# MediaPipe mock
# -----------------------------------------------------------------------
@pytest.fixture
def mock_mediapipe():
    with patch("app.ai.hand_landmarker.hand_landmarker.extract", return_value=[
        {"vector": [0.0] * 63, "handedness": "RIGHT", "raw_landmarks": [0.0] * 63}
    ]):
        yield


# -----------------------------------------------------------------------
# Valid JWT token for tests
# -----------------------------------------------------------------------
@pytest.fixture
def auth_token():
    import jwt as pyjwt
    payload = {
        "sub": "user-test-001",
        "session_id": "sess-001",
        "email_verified": True,
        "iss": settings.USER_JWT_ISSUER,
        "aud": settings.USER_JWT_AUDIENCE,
    }
    return pyjwt.encode(payload, settings.USER_JWT_SECRET_KEY, algorithm=settings.USER_JWT_ALGORITHM)


@pytest.fixture
def auth_headers(auth_token):
    return {"Authorization": f"Bearer {auth_token}"}
