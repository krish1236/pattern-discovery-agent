"""Gap detection tests."""

from src.core.graph import KnowledgeGraph
from src.core.patterns.gaps import detect_gaps
from src.core.types import Edge, EdgeMeta, EdgeType, Node, NodeType


def test_detect_gap_on_recommend_edges() -> None:
    g = KnowledgeGraph()
    meta = EdgeMeta(
        source_tier=1,
        source_family="scholarly",
        source_url="https://x.com",
        source_id="x",
        extraction_confidence=0.9,
        extraction_model="t",
        timestamp="2025-01-01",
        provenance="abstract",
        domain="research",
    )
    target = Node(id="t1", node_type=NodeType.CONCEPT, name="Future work target")
    a1 = Node(
        id="a1",
        node_type=NodeType.ASSERTION,
        name="RecommendationSourceAlpha",
        description="First recommendation text",
    )
    a2 = Node(
        id="a2",
        node_type=NodeType.ASSERTION,
        name="RecommendationSourceBeta",
        description="Second recommendation text",
    )
    g.add_node(target)
    id1 = g.add_node(a1)
    id2 = g.add_node(a2)
    g.add_edge(Edge(id1, target.id, EdgeType.RECOMMENDS, meta))
    g.add_edge(Edge(id2, target.id, EdgeType.RECOMMENDS, meta))

    patterns = detect_gaps(g, min_recommendations=2)
    assert len(patterns) == 1
    assert patterns[0].pattern_type.value == "gap"
