from typing import Optional


def validate_relationship(value: str) -> Optional[str]:
    allowed = {"OWNER", "FAMILY", "FRIEND", "GUEST", "STAFF", "OTHER"}
    if value.upper() not in allowed:
        return f"Invalid relationship: {value}. Allowed: {', '.join(sorted(allowed))}"
    return None


def validate_gesture_type(value: str) -> Optional[str]:
    allowed = {"STATIC", "DYNAMIC"}
    if value.upper() not in allowed:
        return f"Invalid gesture_type: {value}. Allowed: STATIC, DYNAMIC"
    return None


def validate_training_type(value: str) -> Optional[str]:
    allowed = {"OBJECT_DETECTION", "GESTURE_CLASSIFICATION", "FACE_PROFILE_UPDATE", "SCENE_CLASSIFIER"}
    if value.upper() not in allowed:
        return f"Invalid training_type: {value}."
    return None


def validate_dataset_type(value: str) -> Optional[str]:
    allowed = {"OBJECT_DETECTION", "GESTURE", "FACE_SAMPLES", "SCENE"}
    if value.upper() not in allowed:
        return f"Invalid dataset_type: {value}."
    return None
