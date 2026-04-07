"""Bridge detection tests."""

import uuid

import pytest

import src.core.patterns.bridges as bridges_mod
from src.core.graph import KnowledgeGraph
from src.core.patterns.bridges import detect_bridges
from src.core.types import Edge, EdgeMeta, EdgeType, Node, NodeType


@pytest.fixture
def bridge_graph() -> KnowledgeGraph:
    """Two concept cliques + one cross-edge bridge (SOURCE_DOCUMENT endpoints are skipped)."""
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

    # Distinct names required: fuzzy entity resolution merges >0.75 string-similar labels.
    left_names = [str(uuid.uuid4()) for _ in range(4)]
    for i in range(1, 5):
        g.add_node(
            Node(
                id=f"ca{i}",
                node_type=NodeType.CONCEPT,
                name=left_names[i - 1],
                domain="research",
                embedding=[1.0, 0.0, 0.1 * i],
            )
        )
    for i in range(1, 5):
        for j in range(i + 1, 5):
            g.add_edge(
                Edge(
                    source_node_id=f"ca{i}",
                    target_node_id=f"ca{j}",
                    edge_type=EdgeType.CO_OCCURS,
                    meta=meta,
                )
            )

    right_names = [str(uuid.uuid4()) for _ in range(4)]
    for i in range(1, 5):
        g.add_node(
            Node(
                id=f"cb{i}",
                node_type=NodeType.CONCEPT,
                name=right_names[i - 1],
                domain="research",
                embedding=[0.0, 1.0, 0.1 * i],
            )
        )
    for i in range(1, 5):
        for j in range(i + 1, 5):
            g.add_edge(
                Edge(
                    source_node_id=f"cb{i}",
                    target_node_id=f"cb{j}",
                    edge_type=EdgeType.CO_OCCURS,
                    meta=meta,
                )
            )

    n_a1 = g.get_node("ca1")
    n_b1 = g.get_node("cb1")
    assert n_a1 is not None and n_b1 is not None
    # Cross-community cosine must stay within bridge band (above min, below max_obvious).
    n_a1.embedding = [1.0, 0.0, 0.0]
    n_b1.embedding = [0.55, 0.55, 0.3]
    g.add_edge(
        Edge(
            source_node_id="ca1",
            target_node_id="cb1",
            edge_type=EdgeType.CO_OCCURS,
            meta=meta,
        )
    )
    return g


@pytest.fixture
def bridge_graph_with_community_assertions() -> KnowledgeGraph:
    """Two concept clusters + assertions inside each; cross-edge is the bridge."""
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
    # Intra-community similarity high; cross-edge c1–c3 in (min_threshold, max_obvious).
    emb_c1 = [0.7, 0.2, 0.1]
    emb_c2 = [0.65, 0.25, 0.1]
    emb_c3 = [0.2, 0.7, 0.15]
    emb_c4 = [0.25, 0.65, 0.12]
    g.add_node(
        Node(
            id="c1",
            node_type=NodeType.CONCEPT,
            name="QuasarRuntimeKernel",
            domain="research",
            embedding=list(emb_c1),
        )
    )
    g.add_node(
        Node(
            id="c2",
            node_type=NodeType.CONCEPT,
            name="NebulaStorageFabric",
            domain="research",
            embedding=list(emb_c2),
        )
    )
    g.add_edge(Edge(source_node_id="c1", target_node_id="c2", edge_type=EdgeType.SUPPORTS, meta=meta))
    g.add_node(
        Node(
            id="as1",
            node_type=NodeType.ASSERTION,
            name="ZebraAlphaQx",
            description="QuasarRuntimeKernel first community claim text",
            domain="research",
            properties={
                "source_document_id": "doc-a",
                "source_url": "https://a.example/paper",
                "source_tier": 1,
            },
        )
    )
    g.add_edge(Edge(source_node_id="c1", target_node_id="as1", edge_type=EdgeType.SUPPORTS, meta=meta))

    g.add_node(
        Node(
            id="c3",
            node_type=NodeType.CONCEPT,
            name="HelixWorkflowMesh",
            domain="research",
            embedding=list(emb_c3),
        )
    )
    g.add_node(
        Node(
            id="c4",
            node_type=NodeType.CONCEPT,
            name="VertexPolicyGraph",
            domain="research",
            embedding=list(emb_c4),
        )
    )
    g.add_edge(Edge(source_node_id="c3", target_node_id="c4", edge_type=EdgeType.SUPPORTS, meta=meta))
    g.add_node(
        Node(
            id="as2",
            node_type=NodeType.ASSERTION,
            name="MoonBetaYz9",
            description="HelixWorkflowMesh second community claim text",
            domain="research",
            properties={
                "source_document_id": "doc-b",
                "source_url": "https://b.example/paper",
                "source_tier": 2,
            },
        )
    )
    g.add_edge(Edge(source_node_id="c3", target_node_id="as2", edge_type=EdgeType.SUPPORTS, meta=meta))

    g.add_edge(Edge(source_node_id="c1", target_node_id="c3", edge_type=EdgeType.ASSOCIATED_WITH, meta=meta))
    return g


def test_endpoint_significant_token_overlap() -> None:
    assert bridges_mod._endpoint_significant_token_overlap(
        "Natural Language Processing", "Large Language Models"
    )
    assert not bridges_mod._endpoint_significant_token_overlap("LoRA", "Transformer")


class TestBridgeDetection:
    def test_bridge_evidence_includes_community_assertions(
        self, bridge_graph_with_community_assertions: KnowledgeGraph
    ) -> None:
        patterns = detect_bridges(
            bridge_graph_with_community_assertions,
            min_community_size=2,
            semantic_threshold=0.35,
        )
        assert len(patterns) >= 1
        p0 = patterns[0]
        ids = {e.assertion_node_id for e in p0.evidence}
        assert "as1" in ids and "as2" in ids

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

    def test_bridge_evidence_text_mentions_endpoints(
        self, bridge_graph_with_community_assertions: KnowledgeGraph
    ) -> None:
        patterns = detect_bridges(
            bridge_graph_with_community_assertions,
            min_community_size=2,
            semantic_threshold=0.35,
        )
        assert patterns
        p0 = patterns[0]
        blob = " ".join(e.assertion_text.lower() for e in p0.evidence)
        u = p0.details.get("bridge_node_u", "").lower()
        v = p0.details.get("bridge_node_v", "").lower()
        assert u and v
        assert u in blob or any(w in blob for w in u.split() if len(w) > 3)
        assert v in blob or any(w in blob for w in v.split() if len(w) > 3)
