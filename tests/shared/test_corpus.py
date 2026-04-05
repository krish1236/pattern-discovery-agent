"""Corpus manager tests."""

from src.core.types import SourceDocument
from src.shared.corpus import corpus_stats, deduplicate


class TestDedup:
    def test_dedup_by_doi(self) -> None:
        docs = [
            SourceDocument(source_id="a", title="Paper A", metadata={"doi": "10.1234/test"}),
            SourceDocument(source_id="b", title="Paper A copy", metadata={"doi": "10.1234/test"}),
        ]
        result = deduplicate(docs)
        assert len(result) == 1

    def test_dedup_by_title_fuzzy(self) -> None:
        docs = [
            SourceDocument(source_id="a", title="Attention Is All You Need"),
            SourceDocument(source_id="b", title="Attention is All You Need."),
        ]
        result = deduplicate(docs)
        assert len(result) == 1

    def test_keeps_different_papers(self) -> None:
        docs = [
            SourceDocument(source_id="a", title="Paper Alpha", metadata={"doi": "10.1/alpha"}),
            SourceDocument(source_id="b", title="Paper Beta", metadata={"doi": "10.1/beta"}),
        ]
        result = deduplicate(docs)
        assert len(result) == 2

    def test_keeps_richer_duplicate(self) -> None:
        docs = [
            SourceDocument(source_id="a", title="Test", abstract="", metadata={"doi": "10.1/x"}),
            SourceDocument(
                source_id="b",
                title="Test",
                abstract="A full abstract here",
                metadata={"doi": "10.1/x"},
            ),
        ]
        result = deduplicate(docs)
        assert len(result) == 1
        assert result[0].abstract == "A full abstract here"


class TestCorpusStats:
    def test_counts_tiers(self) -> None:
        docs = [
            SourceDocument(source_tier=1, source_family="scholarly"),
            SourceDocument(source_tier=1, source_family="scholarly"),
            SourceDocument(source_tier=2, source_family="code"),
            SourceDocument(source_tier=3, source_family="web"),
        ]
        stats = corpus_stats(docs)
        assert stats["total_documents"] == 4
        assert stats["documents_per_tier"][1] == 2
        assert stats["documents_per_tier"][2] == 1
