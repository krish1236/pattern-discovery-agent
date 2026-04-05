"""Tests for knowledge graph construction."""

import ast
import json
from pathlib import Path

import pytest

from src.core.graph import KnowledgeGraph
from src.core.types import Edge, EdgeMeta, EdgeType, ExtractionResult, Node, NodeType


@pytest.fixture
def empty_graph() -> KnowledgeGraph:
    return KnowledgeGraph()


@pytest.fixture
def sample_meta() -> EdgeMeta:
    return EdgeMeta(
        source_tier=1,
        source_family="scholarly",
        source_url="https://example.com",
        source_id="W123",
        extraction_confidence=0.9,
        extraction_model="haiku",
        timestamp="2025-01-01",
        provenance="abstract",
        domain="research",
    )


class TestGraphConstruction:
    def test_add_node(self, empty_graph: KnowledgeGraph) -> None:
        node = Node(node_type=NodeType.CONCEPT, name="Attention", domain="research")
        nid = empty_graph.add_node(node)
        assert empty_graph.node_count == 1
        assert empty_graph.get_node(nid) is not None
        assert empty_graph.get_node(nid).name == "Attention"

    def test_add_edge(self, empty_graph: KnowledgeGraph, sample_meta: EdgeMeta) -> None:
        n1 = Node(id="n1", node_type=NodeType.ASSERTION, name="Independent alpha")
        n2 = Node(id="n2", node_type=NodeType.ASSERTION, name="Unrelated beta")
        empty_graph.add_node(n1)
        empty_graph.add_node(n2)
        edge = Edge(
            source_node_id="n1",
            target_node_id="n2",
            edge_type=EdgeType.SUPPORTS,
            meta=sample_meta,
        )
        empty_graph.add_edge(edge)
        assert empty_graph.edge_count == 1


class TestEntityResolution:
    def test_merges_same_name_same_type(self, empty_graph: KnowledgeGraph) -> None:
        n1 = Node(node_type=NodeType.CONCEPT, name="BERT", domain="research")
        n2 = Node(node_type=NodeType.CONCEPT, name="bert", domain="research")
        id1 = empty_graph.add_node(n1)
        id2 = empty_graph.add_node(n2)
        assert id1 == id2
        assert empty_graph.node_count == 1

    def test_merges_fuzzy_match(self, empty_graph: KnowledgeGraph) -> None:
        n1 = Node(
            node_type=NodeType.CONCEPT,
            name="Bidirectional Encoder Representations",
            domain="research",
        )
        empty_graph.add_node(n1)
        n2 = Node(node_type=NodeType.CONCEPT, name="BERT", domain="research")
        empty_graph.add_node(n2)
        assert empty_graph.node_count == 2

    def test_does_not_merge_different_types(self, empty_graph: KnowledgeGraph) -> None:
        n1 = Node(node_type=NodeType.CONCEPT, name="Transformer", domain="research")
        n2 = Node(node_type=NodeType.ARTIFACT, name="Transformer", domain="research")
        empty_graph.add_node(n1)
        empty_graph.add_node(n2)
        assert empty_graph.node_count == 2

    def test_never_merges_source_documents(self, empty_graph: KnowledgeGraph) -> None:
        n1 = Node(node_type=NodeType.SOURCE_DOCUMENT, name="Same title")
        n2 = Node(node_type=NodeType.SOURCE_DOCUMENT, name="Same title")
        empty_graph.add_node(n1)
        empty_graph.add_node(n2)
        assert empty_graph.node_count == 2


class TestSerialization:
    def test_roundtrip_json(self, empty_graph: KnowledgeGraph, sample_meta: EdgeMeta) -> None:
        n1 = Node(id="n1", node_type=NodeType.CONCEPT, name="Attention", domain="research")
        n2 = Node(id="n2", node_type=NodeType.ASSERTION, name="Stmt", domain="research")
        empty_graph.add_node(n1)
        empty_graph.add_node(n2)
        edge = Edge(
            source_node_id="n1",
            target_node_id="n2",
            edge_type=EdgeType.ASSOCIATED_WITH,
            meta=sample_meta,
        )
        empty_graph.add_edge(edge)

        json_str = empty_graph.to_json()
        restored = KnowledgeGraph.from_json(json_str)
        assert restored.node_count == 2
        assert restored.edge_count == 1
        assert restored.get_node("n1") is not None
        assert restored.get_node("n1").name == "Attention"

    def test_json_is_valid(self, empty_graph: KnowledgeGraph) -> None:
        n = Node(node_type=NodeType.CONCEPT, name="Test")
        empty_graph.add_node(n)
        json_str = empty_graph.to_json()
        parsed = json.loads(json_str)
        assert "nodes" in parsed
        assert "edges" in parsed

    def test_roundtrip_preserves_embeddings_when_enabled(self, empty_graph: KnowledgeGraph) -> None:
        n = Node(
            id="emb1",
            node_type=NodeType.CONCEPT,
            name="Test",
            domain="research",
            embedding=[0.25, 0.5, 0.75],
        )
        empty_graph.add_node(n)
        restored = KnowledgeGraph.from_json(empty_graph.to_json(include_embeddings=True))
        got = restored.get_node("emb1")
        assert got is not None
        assert got.embedding == [0.25, 0.5, 0.75]

    def test_default_json_omits_embeddings(self, empty_graph: KnowledgeGraph) -> None:
        n = Node(
            id="emb2",
            node_type=NodeType.CONCEPT,
            name="X",
            embedding=[1.0, 2.0],
        )
        empty_graph.add_node(n)
        parsed = json.loads(empty_graph.to_json())
        assert "embedding" not in parsed["nodes"][0]


class TestStats:
    def test_empty_graph_stats(self, empty_graph: KnowledgeGraph) -> None:
        stats = empty_graph.stats()
        assert stats["total_nodes"] == 0
        assert stats["total_edges"] == 0

    def test_stats_count_types(self, empty_graph: KnowledgeGraph) -> None:
        empty_graph.add_node(Node(node_type=NodeType.CONCEPT, name="A"))
        empty_graph.add_node(Node(node_type=NodeType.CONCEPT, name="B"))
        empty_graph.add_node(Node(node_type=NodeType.ACTOR, name="C"))
        stats = empty_graph.stats()
        assert stats["nodes_by_type"]["concept"] == 2
        assert stats["nodes_by_type"]["actor"] == 1


class TestAddExtractionResult:
    def test_adds_nodes_and_edges(self, empty_graph: KnowledgeGraph, sample_meta: EdgeMeta) -> None:
        doc = Node(id="doc1", node_type=NodeType.SOURCE_DOCUMENT, name="Doc")
        assertion = Node(
            id="a1",
            node_type=NodeType.ASSERTION,
            name="Statement",
            description="Statement text",
        )
        edge = Edge(
            source_node_id="doc1",
            target_node_id="a1",
            edge_type=EdgeType.ASSERTS,
            meta=sample_meta,
        )
        result = ExtractionResult(
            source_document_id="doc1",
            nodes=[doc, assertion],
            edges=[edge],
        )
        mapping = empty_graph.add_extraction_result(result)
        assert mapping["doc1"] == "doc1"
        assert mapping["a1"] == "a1"
        assert empty_graph.node_count == 2
        assert empty_graph.edge_count == 1


class TestDomainAgnosticism:
    def test_graph_module_has_no_domain_references(self) -> None:
        graph_path = Path(__file__).resolve().parents[2] / "src" / "core" / "graph.py"
        source = graph_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        banned = {"paper", "claim", "team", "bill", "player", "senator"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                lowered = node.value.lower()
                for term in banned:
                    assert term not in lowered, f"Domain-specific term {term!r} in string literal"
