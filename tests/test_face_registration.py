import pytest
import numpy as np
from unittest.mock import AsyncMock, patch, MagicMock
from io import BytesIO


@pytest.mark.asyncio
async def test_register_face_validates_mime_type(mock_qdrant, mock_minio, mock_ownership_ok):
    from app.services.face_service import face_service
    from tests.conftest import TestSessionLocal
    async with TestSessionLocal() as db:
        with pytest.raises(ValueError, match="Unsupported image type"):
            await face_service.register_face(
                robot_id="robot-001", user_id="user-001",
                display_name="Test", relationship="OWNER",
                image_bytes_list=[b"fakepdf"], content_types=["application/pdf"],
                db=db,
            )


@pytest.mark.asyncio
async def test_register_face_validates_image_size(mock_qdrant, mock_minio, mock_ownership_ok):
    from app.services.face_service import face_service
    from tests.conftest import TestSessionLocal
    big_image = b"x" * (11 * 1024 * 1024)
    async with TestSessionLocal() as db:
        with pytest.raises(ValueError, match="Image too large"):
            await face_service.register_face(
                robot_id="robot-001", user_id="user-001",
                display_name="Test", relationship="OWNER",
                image_bytes_list=[big_image], content_types=["image/jpeg"],
                db=db,
            )


@pytest.mark.asyncio
async def test_register_face_rejects_no_face_detected(mock_qdrant, mock_minio, mock_ownership_ok):
    from app.services.face_service import face_service
    from tests.conftest import TestSessionLocal
    # Patch face_detector.detect to return empty (no face)
    with patch("app.services.face_service.FaceService._extract_embedding", return_value=None):
        async with TestSessionLocal() as db:
            with pytest.raises(ValueError, match="No usable face"):
                await face_service.register_face(
                    robot_id="robot-001", user_id="user-001",
                    display_name="Test", relationship="OWNER",
                    image_bytes_list=[b"fake"], content_types=["image/jpeg"],
                    db=db,
                )


@pytest.mark.asyncio
async def test_list_face_profiles_empty(mock_qdrant):
    from app.services.face_service import face_service
    from tests.conftest import TestSessionLocal
    async with TestSessionLocal() as db:
        profiles = await face_service.list_profiles("robot-new-001", db)
        assert profiles == []


@pytest.mark.asyncio
async def test_get_nonexistent_profile(mock_qdrant):
    from app.services.face_service import face_service
    from tests.conftest import TestSessionLocal
    async with TestSessionLocal() as db:
        profile = await face_service.get_profile("robot-001", "no-such-id", db)
        assert profile is None


@pytest.mark.asyncio
async def test_delete_nonexistent_profile(mock_qdrant, mock_minio):
    from app.services.face_service import face_service
    from tests.conftest import TestSessionLocal
    async with TestSessionLocal() as db:
        result = await face_service.delete_profile("robot-001", "no-such-id", db)
        assert result is False


@pytest.mark.asyncio
async def test_register_face_full_flow(mock_qdrant, mock_minio):
    """Full face registration with mocked InsightFace embedding."""
    import numpy as np
    from app.services.face_service import face_service
    from tests.conftest import TestSessionLocal

    fake_embedding = list(np.random.randn(512).tolist())
    with patch("app.services.face_service.FaceService._extract_embedding", return_value=fake_embedding):
        async with TestSessionLocal() as db:
            profile = await face_service.register_face(
                robot_id="robot-reg-001", user_id="user-001",
                display_name="Thathsara", relationship="OWNER",
                image_bytes_list=[b"fake-image-bytes"], content_types=["image/jpeg"],
                db=db,
            )
            assert profile.display_name == "Thathsara"
            assert profile.relationship == "OWNER"
            assert profile.sample_count == 1
            assert profile.face_profile_id is not None

            # Verify stored in DB
            stored = await face_service.get_profile("robot-reg-001", profile.face_profile_id, db)
            assert stored is not None
            assert stored.display_name == "Thathsara"
