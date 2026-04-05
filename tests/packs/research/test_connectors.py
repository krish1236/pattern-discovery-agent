"""Connector parsing tests (offline)."""

import json
from pathlib import Path

import pytest

from src.core.types import SourceDocument
from src.packs.research.connectors.openalex import _invert_abstract, _parse_work
from src.packs.research.connectors.semantic_scholar import _parse_paper


class TestOpenAlexParsing:
    def test_invert_abstract(self) -> None:
        inverted = {"The": [0], "cat": [1], "sat": [2]}
        assert _invert_abstract(inverted) == "The cat sat"

    def test_invert_abstract_none(self) -> None:
        assert _invert_abstract(None) == ""

    def test_parse_work_returns_source_document(self) -> None:
        work = {
            "id": "https://openalex.org/W123",
            "title": "Test Paper",
            "abstract_inverted_index": {"A": [0], "test": [1]},
            "publication_date": "2024-01-15",
            "cited_by_count": 100,
            "type": "journal-article",
            "authorships": [{"author": {"display_name": "Alice"}}],
            "primary_location": {"is_oa": True},
            "concepts": [{"display_name": "Machine Learning"}],
            "doi": "https://doi.org/10.1234/test",
            "is_oa": True,
        }
        doc = _parse_work(work)
        assert isinstance(doc, SourceDocument)
        assert doc.title == "Test Paper"
        assert doc.abstract == "A test"
        assert doc.source_family == "scholarly"
        assert doc.source_tier == 1
        assert "Alice" in doc.authors


class TestSemanticScholarParsing:
    def test_parse_paper_returns_source_document(self) -> None:
        paper = {
            "paperId": "abc123",
            "title": "Test Paper",
            "abstract": "This is a test abstract",
            "year": 2024,
            "citationCount": 50,
            "influentialCitationCount": 10,
            "authors": [{"name": "Bob"}],
            "externalIds": {"DOI": "10.1234/test"},
            "publicationDate": "2024-03-20",
            "tldr": {"text": "A short summary"},
            "embedding": {"model": "specter2", "vector": [0.1, 0.2, 0.3]},
        }
        doc = _parse_paper(paper)
        assert isinstance(doc, SourceDocument)
        assert doc.title == "Test Paper"
        assert doc.precomputed_embedding == [0.1, 0.2, 0.3]
        assert doc.metadata["tldr"] == "A short summary"
        assert doc.metadata["doi"] == "10.1234/test"

    def test_parse_paper_handles_missing_fields(self) -> None:
        paper = {"paperId": "xyz", "title": "Minimal"}
        doc = _parse_paper(paper)
        assert doc.title == "Minimal"
        assert doc.abstract == ""
        assert doc.precomputed_embedding is None


class TestFixtureFiles:
    def test_openalex_fixture_parses(self) -> None:
        path = Path(__file__).resolve().parents[3] / "fixtures" / "sample_openalex_response.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        work = data["results"][0]
        doc = _parse_work(work)
        assert doc.title == "Fixture work"


class TestConnectorInterface:
    def test_openalex_parse_returns_correct_type(self) -> None:
        work = {"id": "https://openalex.org/W1", "title": "Test"}
        doc = _parse_work(work)
        assert isinstance(doc, SourceDocument)
        assert doc.domain == "research"

    def test_s2_parse_returns_correct_type(self) -> None:
        paper = {"paperId": "abc", "title": "Test"}
        doc = _parse_paper(paper)
        assert isinstance(doc, SourceDocument)
        assert doc.domain == "research"
