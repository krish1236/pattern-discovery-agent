"""Tests for the universal type system."""

import json

import pytest

from src.core.types import (
    BlindSpot,
    ConfidenceLevel,
    Edge,
    EdgeMeta,
    EdgeType,
    EvidenceItem,
    Node,
    NodeType,
    PatternCandidate,
    PatternType,
    SourceDocument,
)


class TestEdgeMeta:
    def test_valid_creation(self) -> None:
        meta = EdgeMeta(
            source_tier=1,
            source_family="scholarly",
            source_url="https://example.com",
            source_id="W123",
            extraction_confidence=0.9,
            extraction_model="claude-haiku-4-5",
            timestamp="2025-01-15",
            provenance="abstract",
            domain="research",
        )
        assert meta.source_tier == 1
        assert meta.extraction_confidence == 0.9

    def test_invalid_tier_raises(self) -> None:
        with pytest.raises(ValueError, match="source_tier must be 1-4"):
            EdgeMeta(
                source_tier=5,
                source_family="scholarly",
                source_url="",
                source_id="",
                extraction_confidence=0.5,
                extraction_model="",
                timestamp="",
                provenance="",
                domain="",
            )

    def test_invalid_confidence_raises(self) -> None:
        with pytest.raises(ValueError, match="extraction_confidence must be 0.0-1.0"):
            EdgeMeta(
                source_tier=1,
                source_family="scholarly",
                source_url="",
                source_id="",
                extraction_confidence=1.5,
                extraction_model="",
                timestamp="",
                provenance="",
                domain="",
            )

    def test_roundtrip_json(self) -> None:
        meta = EdgeMeta(
            source_tier=2,
            source_family="code",
            source_url="https://github.com/x",
            source_id="repo_123",
            extraction_confidence=0.85,
            extraction_model="api_metadata",
            timestamp="2025-06-01",
            provenance="metadata",
            domain="research",
        )
        d = meta.to_dict()
        restored = EdgeMeta.from_dict(d)
        assert restored.source_tier == meta.source_tier
        assert restored.source_url == meta.source_url
        assert restored.extraction_confidence == meta.extraction_confidence


class TestNode:
    def test_auto_generates_id(self) -> None:
        node = Node(node_type=NodeType.ASSERTION, name="test")
        assert len(node.id) == 36

    def test_roundtrip_json(self) -> None:
        node = Node(
            node_type=NodeType.ACTOR,
            name="Test Lab",
            description="A research lab",
            domain="research",
            properties={"affiliation": "MIT"},
        )
        d = node.to_dict()
        json_str = json.dumps(d)
        restored = Node.from_dict(json.loads(json_str))
        assert restored.name == node.name
        assert restored.node_type == NodeType.ACTOR
        assert restored.properties["affiliation"] == "MIT"

    def test_embedding_not_serialized(self) -> None:
        node = Node(
            node_type=NodeType.CONCEPT,
            name="attention",
            embedding=[0.1, 0.2, 0.3],
        )
        d = node.to_dict()
        assert "embedding" not in d

    def test_embedding_serializes_when_requested(self) -> None:
        node = Node(
            node_type=NodeType.CONCEPT,
            name="attention",
            embedding=[0.1, 0.2, 0.3],
        )
        d = node.to_dict(include_embedding=True)
        assert d["embedding"] == [0.1, 0.2, 0.3]
        restored = Node.from_dict(d)
        assert restored.embedding == [0.1, 0.2, 0.3]


class TestEdge:
    def test_roundtrip_json(self) -> None:
        meta = EdgeMeta(
            source_tier=1,
            source_family="scholarly",
            source_url="https://example.com",
            source_id="W123",
            extraction_confidence=0.9,
            extraction_model="claude-haiku-4-5",
            timestamp="2025-01-15",
            provenance="abstract",
            domain="research",
        )
        edge = Edge(
            source_node_id="node_1",
            target_node_id="node_2",
            edge_type=EdgeType.SUPPORTS,
            meta=meta,
        )
        d = edge.to_dict()
        json_str = json.dumps(d)
        restored = Edge.from_dict(json.loads(json_str))
        assert restored.edge_type == EdgeType.SUPPORTS
        assert restored.meta.source_tier == 1


class TestSourceDocument:
    def test_roundtrip_json(self) -> None:
        doc = SourceDocument(
            source_id="W2741809807",
            source_family="scholarly",
            source_url="https://openalex.org/W2741809807",
            title="Attention Is All You Need",
            abstract="We propose...",
            authors=["Vaswani, A."],
            publication_date="2017-06-12",
            source_tier=1,
            domain="research",
            metadata={"citation_count": 90000},
        )
        d = doc.to_dict()
        json_str = json.dumps(d)
        restored = SourceDocument.from_dict(json.loads(json_str))
        assert restored.title == doc.title
        assert restored.source_tier == 1

    def test_precomputed_embedding_not_serialized(self) -> None:
        doc = SourceDocument(title="test", precomputed_embedding=[0.1, 0.2])
        d = doc.to_dict()
        assert "precomputed_embedding" not in d


class TestPatternCandidate:
    def test_evidence_count(self) -> None:
        evidence = [
            EvidenceItem(
                assertion_node_id="a1",
                assertion_text="claim 1",
                source_document_id="d1",
                source_url="https://a.com",
                source_tier=1,
            ),
            EvidenceItem(
                assertion_node_id="a2",
                assertion_text="claim 2",
                source_document_id="d2",
                source_url="https://b.com",
                source_tier=2,
            ),
        ]
        pattern = PatternCandidate(
            pattern_type=PatternType.BRIDGE,
            title="Test bridge",
            evidence=evidence,
        )
        assert pattern.evidence_count == 2
        assert pattern.source_tiers == {1, 2}
        assert len(pattern.source_urls) == 2

    def test_serialization(self) -> None:
        pattern = PatternCandidate(
            pattern_type=PatternType.CONTRADICTION,
            title="Test contradiction",
            confidence_level=ConfidenceLevel.MEDIUM,
            domain="research",
        )
        d = pattern.to_dict()
        assert d["pattern_type"] == "contradiction"
        assert d["confidence_level"] == "medium"


class TestDomainAgnosticism:
    def test_node_types_are_generic(self) -> None:
        for nt in NodeType:
            assert "paper" not in nt.value
            assert "claim" not in nt.value
            assert "team" not in nt.value
            assert "bill" not in nt.value

    def test_edge_types_are_generic(self) -> None:
        for et in EdgeType:
            assert "cites" not in et.value
            assert "authored" not in et.value
