"""Gap detection tests."""

import uuid

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
    target = Node(id="t1", node_type=NodeType.CONCEPT, name=f"GapTargetConcept {uuid.uuid4()}")
    n1 = str(uuid.uuid4())
    n2 = str(uuid.uuid4())
    a1 = Node(
        id="a1",
        node_type=NodeType.ASSERTION,
        name=n1,
        description=f"First recommendation {uuid.uuid4()}",
        properties={"source_document_id": "doc-a"},
    )
    a2 = Node(
        id="a2",
        node_type=NodeType.ASSERTION,
        name=n2,
        description=f"Second recommendation {uuid.uuid4()}",
        properties={"source_document_id": "doc-b"},
    )
    g.add_node(target)
    id1 = g.add_node(a1)
    id2 = g.add_node(a2)
    g.add_edge(Edge(id1, target.id, EdgeType.RECOMMENDS, meta))
    g.add_edge(Edge(id2, target.id, EdgeType.RECOMMENDS, meta))

    patterns = detect_gaps(g, min_recommendations=2)
    assert len(patterns) == 1
    assert patterns[0].pattern_type.value == "gap"


def test_gap_excludes_actors() -> None:
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
    target = Node(id="actor1", node_type=NodeType.ACTOR, name="VendorCo")
    a1 = Node(
        id="r1",
        node_type=NodeType.ASSERTION,
        name="R1",
        properties={"source_document_id": "d1"},
    )
    a2 = Node(
        id="r2",
        node_type=NodeType.ASSERTION,
        name="R2",
        properties={"source_document_id": "d2"},
    )
    g.add_node(target)
    g.add_node(a1)
    g.add_node(a2)
    g.add_edge(Edge("r1", "actor1", EdgeType.RECOMMENDS, meta))
    g.add_edge(Edge("r2", "actor1", EdgeType.RECOMMENDS, meta))
    assert detect_gaps(g, min_recommendations=2) == []


def test_gap_requires_distinct_source_documents() -> None:
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
    target = Node(id="t1", node_type=NodeType.CONCEPT, name="Understudied topic")
    a1 = Node(
        id="a1",
        node_type=NodeType.ASSERTION,
        name="A1",
        properties={"source_document_id": "same-doc"},
    )
    a2 = Node(
        id="a2",
        node_type=NodeType.ASSERTION,
        name="A2",
        properties={"source_document_id": "same-doc"},
    )
    g.add_node(target)
    g.add_node(a1)
    g.add_node(a2)
    g.add_edge(Edge("a1", "t1", EdgeType.RECOMMENDS, meta))
    g.add_edge(Edge("a2", "t1", EdgeType.RECOMMENDS, meta))
    assert detect_gaps(g, min_recommendations=2) == []
