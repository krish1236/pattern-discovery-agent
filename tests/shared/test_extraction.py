"""Extraction pipeline tests (no live LLM)."""

from src.core.types import EdgeType, NodeType, SourceDocument
from src.packs.research.schema import get_research_schema
from src.shared.extraction import _parse_extraction


class TestParseExtraction:
    def setup_method(self) -> None:
        self.schema = get_research_schema()
        self.doc = SourceDocument(
            id="doc_1",
            source_id="W123",
            source_family="scholarly",
            source_url="https://example.com",
            title="Test Paper",
            abstract="A test abstract",
            source_tier=1,
            domain="research",
        )

    def test_parses_assertions(self) -> None:
        raw = {
            "assertions": [{"text": "Transformers are better than RNNs", "polarity": "positive"}],
            "entities": [],
            "relationships": [],
        }
        result = _parse_extraction(raw, self.doc, self.schema)
        assertion_nodes = [n for n in result.nodes if n.node_type == NodeType.ASSERTION]
        assert len(assertion_nodes) == 1
        assert "Transformers" in assertion_nodes[0].description
        assert assertion_nodes[0].properties.get("source_document_id") == "doc_1"

    def test_parses_entities(self) -> None:
        raw = {
            "assertions": [],
            "entities": [{"name": "BERT", "entity_type": "concept", "description": "A language model"}],
            "relationships": [],
        }
        result = _parse_extraction(raw, self.doc, self.schema)
        concept_nodes = [n for n in result.nodes if n.node_type == NodeType.CONCEPT]
        assert len(concept_nodes) == 1
        assert concept_nodes[0].name == "BERT"

    def test_creates_source_document_node(self) -> None:
        raw = {"assertions": [], "entities": [], "relationships": []}
        result = _parse_extraction(raw, self.doc, self.schema)
        doc_nodes = [n for n in result.nodes if n.node_type == NodeType.SOURCE_DOCUMENT]
        assert len(doc_nodes) == 1

    def test_links_assertions_to_document(self) -> None:
        raw = {
            "assertions": [{"text": "Test claim", "polarity": "neutral"}],
            "entities": [],
            "relationships": [],
        }
        result = _parse_extraction(raw, self.doc, self.schema)
        asserts_edges = [e for e in result.edges if e.edge_type == EdgeType.ASSERTS]
        assert len(asserts_edges) == 1

    def test_handles_empty_extraction(self) -> None:
        raw = {"assertions": [], "entities": [], "relationships": []}
        result = _parse_extraction(raw, self.doc, self.schema)
        assert result.source_document_id == "doc_1"
        assert len(result.nodes) == 1

    def test_edges_carry_full_metadata(self) -> None:
        raw = {
            "assertions": [{"text": "Test claim"}],
            "entities": [],
            "relationships": [],
        }
        result = _parse_extraction(raw, self.doc, self.schema)
        for edge in result.edges:
            assert edge.meta.source_tier == 1
            assert edge.meta.source_family == "scholarly"
            assert edge.meta.domain == "research"
