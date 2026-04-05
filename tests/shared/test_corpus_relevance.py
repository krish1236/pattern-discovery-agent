"""Corpus relevance filter."""

from unittest.mock import patch

import numpy as np

from src.core.types import SourceDocument
from src.shared.corpus import filter_by_relevance


@patch("src.shared.embeddings.embed_texts")
def test_filter_by_relevance_drops_off_topic(mock_embed) -> None:
    """Topic aligned with doc0 embedding; doc1 orthogonal -> dropped."""

    def _fake(texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), 4), dtype=np.float64)
        for i, t in enumerate(texts):
            tl = t.lower()
            if "bitcoin" in tl or "scaling" in tl:
                out[i, 0] = 1.0
            elif "macrophage" in tl or "cell" in tl:
                out[i, 1] = 1.0
            else:
                out[i, 2] = 1.0
        return out

    mock_embed.side_effect = _fake

    docs = [
        SourceDocument(
            source_id="1",
            source_family="scholarly",
            title="Bitcoin scaling analysis",
            abstract="Bitcoin layer-two scaling rollup discussion",
        ),
        SourceDocument(
            source_id="2",
            source_family="scholarly",
            title="Macrophage plasticity",
            abstract="Immune cell macrophage plasticity in tissue",
        ),
    ]
    topic = "Bitcoin layer-2 scaling"
    kept = filter_by_relevance(docs, topic, min_similarity=0.25)
    titles = {d.title for d in kept}
    assert "Bitcoin scaling analysis" in titles
    assert "Macrophage plasticity" not in titles
