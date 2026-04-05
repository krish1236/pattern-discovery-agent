"""Drift detection tests."""

import uuid
from unittest.mock import patch

import numpy as np

from src.core.graph import KnowledgeGraph
from src.core.patterns.drift import detect_drift
from src.core.types import Edge, EdgeMeta, EdgeType, Node, NodeType


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


@patch("src.core.patterns.drift.embed_texts")
def test_drift_produces_evidence_for_assertions(mock_embed) -> None:
    call = [0]

    def _emb(texts: list[str]) -> np.ndarray:
        call[0] += 1
        n = len(texts)
        if call[0] == 1:
            e = np.zeros((n, 8), dtype=np.float64)
            e[:3, 0] = 1.0
            e[3:, 1] = 1.0
            return e
        e = np.zeros((n, 8), dtype=np.float64)
        for i in range(n):
            e[i, i % 3] = 1.0
        return e

    mock_embed.side_effect = _emb

    g = KnowledgeGraph()
    meta = EdgeMeta(
        source_tier=1,
        source_family="scholarly",
        source_url="",
        source_id="",
        extraction_confidence=0.9,
        extraction_model="t",
        timestamp="2025-01-01",
        provenance="t",
        domain="research",
    )
    for i in range(3):
        dn = str(uuid.uuid4())
        an = str(uuid.uuid4())
        did = f"d22_{i}"
        aid = f"a22_{i}"
        g.add_node(
            Node(
                id=did,
                node_type=NodeType.SOURCE_DOCUMENT,
                name=dn,
                description=f"doc 2022 {i} {uuid.uuid4()}",
                properties={"publication_date": "2022-01-01"},
            )
        )
        g.add_node(
            Node(
                id=aid,
                node_type=NodeType.ASSERTION,
                name=an,
                description=f"claim alpha {i} drift {uuid.uuid4()}",
                properties={"publication_date": "2022-06-01"},
            )
        )
        g.add_edge(Edge(did, aid, EdgeType.ASSERTS, meta))
    for i in range(3):
        dn = str(uuid.uuid4())
        an = str(uuid.uuid4())
        did = f"d23_{i}"
        aid = f"a23_{i}"
        g.add_node(
            Node(
                id=did,
                node_type=NodeType.SOURCE_DOCUMENT,
                name=dn,
                description=f"doc 2023 {i} {uuid.uuid4()}",
                properties={"publication_date": "2023-01-01"},
            )
        )
        g.add_node(
            Node(
                id=aid,
                node_type=NodeType.ASSERTION,
                name=an,
                description=f"claim beta {i} drift {uuid.uuid4()}",
                properties={"publication_date": "2023-06-01"},
            )
        )
        g.add_edge(Edge(did, aid, EdgeType.ASSERTS, meta))

    patterns = detect_drift(g, min_cluster_size=2, min_windows=2)
    assert patterns
    assert any(len(p.evidence) > 0 for p in patterns)
