"""Drift detection tests."""

from unittest.mock import patch

import numpy as np

from src.core.graph import KnowledgeGraph
from src.core.patterns.drift import detect_drift
from src.core.types import Node, NodeType


def test_drift_empty_without_dates() -> None:
    g = KnowledgeGraph()
    g.add_node(Node(node_type=NodeType.CONCEPT, name="No date"))
    assert detect_drift(g) == []


@patch("src.core.patterns.drift.embed_texts")
def test_drift_runs_with_mocked_embeddings(mock_embed) -> None:
    mock_embed.return_value = np.random.RandomState(0).randn(5, 8)

    g = KnowledgeGraph()
    for i in range(5):
        g.add_node(
            Node(
                id=f"y22_{i}",
                node_type=NodeType.SOURCE_DOCUMENT,
                name=f"D22-{i}",
                description=f"topic alpha {i}",
                properties={"publication_date": "2022-06-01"},
            )
        )
    for i in range(5):
        g.add_node(
            Node(
                id=f"y23_{i}",
                node_type=NodeType.SOURCE_DOCUMENT,
                name=f"D23-{i}",
                description=f"topic beta {i}",
                properties={"publication_date": "2023-06-01"},
            )
        )

    patterns = detect_drift(g, min_cluster_size=3, min_windows=2)
    assert isinstance(patterns, list)
