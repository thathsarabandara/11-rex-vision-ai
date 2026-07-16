import numpy as np

LANDMARK_DIM = 63  # 21 * 3


def normalize_hand_landmarks(raw_coords: list[float]) -> list[float]:
    """Centre on wrist and scale to unit range."""
    arr = np.array(raw_coords, dtype=np.float32).reshape(21, 3)
    arr -= arr[0]  # centre on wrist
    scale = np.max(np.abs(arr))
    if scale > 0:
        arr /= scale
    return arr.flatten().tolist()


def landmarks_to_feature_vector(landmarks: list[dict]) -> list[float]:
    """Convert a list of {x, y, z} dicts to a flat normalised vector."""
    coords = []
    for lm in landmarks:
        coords.extend([lm.get("x", 0.0), lm.get("y", 0.0), lm.get("z", 0.0)])
    return normalize_hand_landmarks(coords)
