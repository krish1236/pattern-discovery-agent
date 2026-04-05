"""Extraction pipeline tests (no live LLM)."""

import pytest

from src.core.types import EdgeType, NodeType, SourceDocument
from src.packs.research.schema import get_research_schema
from src.shared.extraction import (
    _parse_extraction,
    _parse_extraction_json_payload,
    chunk_documents_for_extraction,
)


def test_parse_extraction_json_payload_strips_markdown_fence() -> None:
    raw = """```json
[{"assertions": [], "entities": [], "relationships": []}]
```"""
    out = _parse_extraction_json_payload(raw)
    assert len(out) == 1
    assert out[0]["assertions"] == []


def test_chunk_documents_for_extraction_respects_size_and_count() -> None:
    import src.shared.extraction as ext

    prev_bs, prev_chars = ext.MAX_BATCH_SIZE, ext.MAX_BATCH_TEXT_CHARS
    try:
        ext.MAX_BATCH_SIZE = 3
        ext.MAX_BATCH_TEXT_CHARS = 100
        docs = [
            SourceDocument(title="a", abstract="x" * 40, source_id="1"),
            SourceDocument(title="b", abstract="y" * 40, source_id="2"),
            SourceDocument(title="c", abstract="z" * 40, source_id="3"),
            SourceDocument(title="d", abstract="w" * 40, source_id="4"),
        ]
        batches = chunk_documents_for_extraction(docs)
        assert len(batches) >= 2
        assert sum(len(b) for b in batches) == 4
        for b in batches:
            assert len(b) <= 3
    finally:
        ext.MAX_BATCH_SIZE = prev_bs
        ext.MAX_BATCH_TEXT_CHARS = prev_chars


def test_parse_extraction_json_payload_strips_double_fenced_payload() -> None:
    raw = """```\n```json
[{"assertions": [], "entities": [], "relationships": []}]
```"""
    out = _parse_extraction_json_payload(raw)
    assert len(out) == 1


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
        doc_to_entity = [
            e
            for e in result.edges
            if e.edge_type == EdgeType.ASSOCIATED_WITH
            and e.source_node_id == self.doc.id
            and e.target_node_id == concept_nodes[0].id
        ]
        assert len(doc_to_entity) == 1

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

    def test_future_work_language_adds_recommends_edges(self) -> None:
        raw = {
            "assertions": [
                {
                    "text": "Future work should explore transformer architectures in more detail.",
                    "polarity": "neutral",
                }
            ],
            "entities": [{"name": "transformer architectures", "entity_type": "concept", "description": ""}],
            "relationships": [],
        }
        result = _parse_extraction(raw, self.doc, self.schema)
        rec = [e for e in result.edges if e.edge_type == EdgeType.RECOMMENDS]
        assert len(rec) >= 1

    def test_precomputed_embedding_copied_to_source_document_node(self) -> None:
        vec = [0.25, -0.5, 0.125]
        doc = SourceDocument(
            id="doc_emb",
            source_id="W1",
            source_family="scholarly",
            source_url="https://example.com",
            title="T",
            abstract="Abstract text here.",
            source_tier=1,
            domain="research",
            precomputed_embedding=vec,
        )
        raw = {"assertions": [], "entities": [], "relationships": []}
        result = _parse_extraction(raw, doc, self.schema)
        src = next(n for n in result.nodes if n.node_type == NodeType.SOURCE_DOCUMENT)
        assert src.embedding == vec
        assert src.embedding is not doc.precomputed_embedding


@pytest.mark.asyncio
async def test_extract_batch_empty_skips_llm() -> None:
    from src.packs.research.schema import get_research_schema
    from src.shared.extraction import extract_batch

    schema = get_research_schema()
    out = await extract_batch([], schema)
    assert out == []
