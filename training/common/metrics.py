"""Shared evaluation metric helpers."""
from typing import Optional
import numpy as np


def precision_recall_f1(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def percentile_latency(latencies_ms: list[float], percentile: float = 95.0) -> float:
    if not latencies_ms:
        return 0.0
    return float(np.percentile(latencies_ms, percentile))


def format_confusion_matrix(matrix: list[list[int]], labels: list[str]) -> dict:
    return {"labels": labels, "matrix": matrix}
