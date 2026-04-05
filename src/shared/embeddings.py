"""Text embeddings (sentence-transformers) and cosine similarity."""

from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_model: Any = None
MAX_EMBEDDING_CACHE = 8192
_embedding_cache: OrderedDict[str, np.ndarray] = OrderedDict()


def clear_embedding_cache() -> None:
    """Clear the in-process embedding cache (mainly for tests)."""
    _embedding_cache.clear()


def _cache_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _cache_get(key: str) -> np.ndarray | None:
    val = _embedding_cache.get(key)
    if val is not None:
        _embedding_cache.move_to_end(key)
    return val


def _cache_put(key: str, vec: np.ndarray) -> None:
    _embedding_cache[key] = vec
    _embedding_cache.move_to_end(key)
    while len(_embedding_cache) > MAX_EMBEDDING_CACHE:
        _embedding_cache.popitem(last=False)


def _get_model() -> Any:
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Loaded embedding model: all-MiniLM-L6-v2")
    return _model


def embed_texts(texts: list[str], batch_size: int = 64) -> np.ndarray:
    if not texts:
        return np.array([])

    keys = [_cache_key(t) for t in texts]
    out: list[np.ndarray | None] = [None] * len(texts)
    pending_idx: list[int] = []
    pending_texts: list[str] = []

    for i, k in enumerate(keys):
        hit = _cache_get(k)
        if hit is not None:
            out[i] = np.asarray(hit)
        else:
            pending_idx.append(i)
            pending_texts.append(texts[i])

    if pending_texts:
        model = _get_model()
        computed = model.encode(
            pending_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        arr = np.asarray(computed)
        for j, orig_i in enumerate(pending_idx):
            row = np.asarray(arr[j])
            _cache_put(keys[orig_i], row)
            out[orig_i] = row

    return np.stack([np.asarray(x) for x in out], axis=0)


def embed_single(text: str) -> list[float]:
    return embed_texts([text])[0].tolist()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normalized = embeddings / norms
    return np.dot(normalized, normalized.T)
