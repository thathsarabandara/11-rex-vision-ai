import numpy as np
from typing import Optional


def cosine_similarity(a: list[float], b: list[float]) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def normalize_vector(v: list[float]) -> list[float]:
    arr = np.array(v, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm == 0:
        return v
    return (arr / norm).tolist()


def average_embeddings(embeddings: list[list[float]]) -> list[float]:
    if not embeddings:
        return []
    arr = np.mean(np.array(embeddings, dtype=np.float32), axis=0)
    return normalize_vector(arr.tolist())
