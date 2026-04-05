"""Embedding helpers (model load mocked)."""

from unittest.mock import MagicMock, patch

import numpy as np

import pytest


def test_cosine_similarity() -> None:
    from src.shared.embeddings import cosine_similarity

    a = np.array([1.0, 0.0])
    b = np.array([1.0, 0.0])
    assert abs(cosine_similarity(a, b) - 1.0) < 1e-6


def test_cosine_similarity_zero_norm() -> None:
    from src.shared.embeddings import cosine_similarity

    assert cosine_similarity(np.zeros(2), np.array([1.0, 0.0])) == 0.0


@pytest.fixture(autouse=True)
def reset_embedding_model() -> None:
    import src.shared.embeddings as emb

    emb._model = None
    emb.clear_embedding_cache()
    yield
    emb._model = None
    emb.clear_embedding_cache()


@patch("sentence_transformers.SentenceTransformer")
def test_embed_texts_cache_hits_skip_second_encode(mock_st_class) -> None:
    import src.shared.embeddings as emb

    inst = MagicMock()
    inst.encode.return_value = np.ones((1, 5))
    mock_st_class.return_value = inst
    emb._model = None
    emb.clear_embedding_cache()
    emb.embed_texts(["same text twice"])
    emb.embed_texts(["same text twice"])
    assert inst.encode.call_count == 1


@patch("sentence_transformers.SentenceTransformer")
def test_embed_texts_uses_sentence_transformer(mock_st_class) -> None:
    import src.shared.embeddings as emb

    inst = MagicMock()
    inst.encode.return_value = np.ones((3, 5))
    mock_st_class.return_value = inst

    emb._model = None
    out = emb.embed_texts(["a", "b", "c"])
    assert out.shape == (3, 5)
    inst.encode.assert_called_once()
    emb._model = None
