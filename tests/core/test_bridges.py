"""Bridge detection tests."""

import pytest

from src.core.graph import KnowledgeGraph
from src.core.patterns.bridges import detect_bridges
from src.core.types import Edge, EdgeMeta, EdgeType, Node, NodeType


@pytest.fixture
def bridge_graph() -> KnowledgeGraph:
    g = KnowledgeGraph()
    meta = EdgeMeta(
        source_tier=1,
        source_family="scholarly",
        source_url="",
        source_id="",
        extraction_confidence=0.9,
        extraction_model="test",
        timestamp="2025-01-01",
        provenance="test",
        domain="research",
    )

    for i in range(1, 5):
        g.add_node(
            Node(
                id=f"a{i}",
                node_type=NodeType.SOURCE_DOCUMENT,
                name=f"UniqueSourceAlpha-{i}-title",
                domain="research",
                embedding=[1.0, 0.0, 0.1 * i],
            )
        )
    for i in range(1, 5):
        for j in range(i + 1, 5):
            g.add_edge(
                Edge(
                    source_node_id=f"a{i}",
                    target_node_id=f"a{j}",
                    edge_type=EdgeType.CO_OCCURS,
                    meta=meta,
                )
            )

    for i in range(1, 5):
        g.add_node(
            Node(
                id=f"b{i}",
                node_type=NodeType.SOURCE_DOCUMENT,
                name=f"UniqueSourceBeta-{i}-title",
                domain="research",
                embedding=[0.0, 1.0, 0.1 * i],
            )
        )
    for i in range(1, 5):
        for j in range(i + 1, 5):
            g.add_edge(
                Edge(
                    source_node_id=f"b{i}",
                    target_node_id=f"b{j}",
                    edge_type=EdgeType.CO_OCCURS,
                    meta=meta,
                )
            )

    n_a1 = g.get_node("a1")
    n_b1 = g.get_node("b1")
    assert n_a1 is not None and n_b1 is not None
    n_a1.embedding = [0.5, 0.5, 0.1]
    n_b1.embedding = [0.5, 0.5, 0.2]
    g.add_edge(
        Edge(
            source_node_id="a1",
            target_node_id="b1",
            edge_type=EdgeType.CO_OCCURS,
            meta=meta,
        )
    )
    return g


class TestBridgeDetection:
    def test_finds_bridge_in_synthetic_graph(self, bridge_graph: KnowledgeGraph) -> None:
        patterns = detect_bridges(bridge_graph, min_community_size=2, semantic_threshold=0.3)
        assert len(patterns) >= 1
        assert patterns[0].pattern_type.value == "bridge"
        assert patterns[0].details["semantic_similarity"] > 0.3

    def test_returns_empty_for_small_graph(self) -> None:
        g = KnowledgeGraph()
        g.add_node(Node(node_type=NodeType.CONCEPT, name="Only one"))
        assert detect_bridges(g) == []

    def test_pattern_shape(self, bridge_graph: KnowledgeGraph) -> None:
        patterns = detect_bridges(bridge_graph, min_community_size=2)
        for p in patterns:
            assert hasattr(p, "pattern_type")
            assert hasattr(p, "evidence")
            assert hasattr(p, "blind_spots")
