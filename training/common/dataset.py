"""Common dataset utilities for training scripts."""
import os
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def load_yolo_dataset(data_yaml: str) -> dict:
    """Load and validate a YOLO data.yaml config."""
    try:
        import yaml  # type: ignore
        with open(data_yaml) as f:
            data = yaml.safe_load(f)
        required = {"train", "val", "names"}
        missing = required - set(data.keys())
        if missing:
            raise ValueError(f"data.yaml missing keys: {missing}")
        return data
    except Exception as exc:
        raise ValueError(f"Invalid YOLO dataset: {exc}") from exc


def count_images(directory: str, extensions: tuple = (".jpg", ".jpeg", ".png")) -> int:
    """Count images in a directory recursively."""
    return sum(1 for f in Path(directory).rglob("*") if f.suffix.lower() in extensions)


def validate_gesture_csv(path: str) -> Optional[str]:
    """Return error message or None if gesture CSV is valid."""
    try:
        import pandas as pd  # type: ignore
        df = pd.read_csv(path)
        if "gesture_name" not in df.columns:
            return "Missing 'gesture_name' column"
        if len(df.columns) != 64:  # 1 label + 63 features
            return f"Expected 64 columns (1 label + 63 features), got {len(df.columns)}"
        return None
    except Exception as exc:
        return str(exc)
