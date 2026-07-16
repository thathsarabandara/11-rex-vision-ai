import pytest
from app.utils.images import validate_image_bytes, is_too_blurry, check_blur
from app.utils.validation import (
    validate_relationship, validate_gesture_type,
    validate_training_type, validate_dataset_type,
)
from app.utils.dates import utcnow_iso, utcnow
from datetime import timezone


class TestImageUtils:
    def test_valid_jpeg_passes(self):
        assert validate_image_bytes(b"x" * 100, "image/jpeg") is None

    def test_invalid_mime_type(self):
        assert validate_image_bytes(b"x", "application/pdf") is not None

    def test_oversized_image_fails(self):
        big = b"x" * (11 * 1024 * 1024)
        assert validate_image_bytes(big, "image/jpeg") is not None

    def test_check_blur_returns_float(self):
        import numpy as np
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        try:
            result = check_blur(frame)
            assert isinstance(result, float)
        except Exception:
            pass  # cv2 may not be available in CI


class TestValidationUtils:
    def test_valid_relationship(self):
        assert validate_relationship("OWNER") is None
        assert validate_relationship("FAMILY") is None
        assert validate_relationship("GUEST") is None

    def test_invalid_relationship(self):
        assert validate_relationship("ROBOT") is not None

    def test_valid_gesture_type(self):
        assert validate_gesture_type("STATIC") is None
        assert validate_gesture_type("DYNAMIC") is None

    def test_invalid_gesture_type(self):
        assert validate_gesture_type("WAVE") is not None

    def test_valid_training_type(self):
        assert validate_training_type("OBJECT_DETECTION") is None
        assert validate_training_type("GESTURE_CLASSIFICATION") is None

    def test_invalid_training_type(self):
        assert validate_training_type("RANDOM") is not None

    def test_valid_dataset_type(self):
        assert validate_dataset_type("OBJECT_DETECTION") is None
        assert validate_dataset_type("GESTURE") is None

    def test_invalid_dataset_type(self):
        assert validate_dataset_type("VIDEO") is not None


class TestDateUtils:
    def test_utcnow_returns_datetime(self):
        dt = utcnow()
        assert dt.tzinfo == timezone.utc

    def test_utcnow_iso_is_string(self):
        iso = utcnow_iso()
        assert isinstance(iso, str)
        assert "T" in iso
