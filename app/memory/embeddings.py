"""Lightweight, dependency-free text embeddings: hashed character n-grams,
L2-normalized. Deterministic, fast, works on any laptop. Good enough for
similarity lookup; swap in sentence-transformers on the GPU box later by
replacing embed() - the rest of the system only sees vectors."""

import hashlib
import numpy as np

DIM = 256


def embed(text: str) -> np.ndarray:
    vec = np.zeros(DIM, dtype=np.float32)
    t = " " + text.lower().strip() + " "
    for n in (3, 4, 5):
        for i in range(len(t) - n + 1):
            gram = t[i:i + n]
            h = int(hashlib.md5(gram.encode()).hexdigest()[:8], 16)
            vec[h % DIM] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def to_bytes(v: np.ndarray) -> bytes:
    return v.astype(np.float32).tobytes()


def from_bytes(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32)