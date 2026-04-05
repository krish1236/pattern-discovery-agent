"""Checkpoint blob loading (no storage I/O)."""

import json

from src.checkpoint import load_checkpoint_from_blobs
from src.core.graph import KnowledgeGraph
from src.core.types import ExtractionResult, SourceDocument


def test_load_checkpoint_disabled_without_resume() -> None:
    store = {"graph.json": b"{}"}

    def get_file(key: str) -> bytes | None:
        return store.get(key)

    assert load_checkpoint_from_blobs(get_file, resume=False) is None


def test_load_checkpoint_requires_all_blobs() -> None:
    store = {"graph.json": b"{}"}

    def get_file(key: str) -> bytes | None:
        return store.get(key)

    assert load_checkpoint_from_blobs(get_file, resume=True) is None


def test_load_checkpoint_round_trip_minimal() -> None:
    doc = SourceDocument(
        id="doc-1",
        source_id="s1",
        source_family="scholarly",
        source_url="https://example.com",
        title="Paper",
        abstract="Abstract",
        source_tier=1,
        domain="research",
    )
    results = [ExtractionResult(source_document_id="doc-1", nodes=[], edges=[])]
    graph = KnowledgeGraph()

    store = {
        "documents.json": json.dumps([doc.to_dict()]).encode(),
        "extraction_results.json": json.dumps(
            [
                {
                    "doc_id": r.source_document_id,
                    "nodes": [n.to_dict() for n in r.nodes],
                    "edges": [e.to_dict() for e in r.edges],
                }
                for r in results
            ]
        ).encode(),
        "graph.json": graph.to_json().encode(),
    }

    def get_file(key: str) -> bytes | None:
        return store.get(key)

    loaded = load_checkpoint_from_blobs(get_file, resume=True)
    assert loaded is not None
    docs, ext, g = loaded
    assert len(docs) == 1
    assert docs[0].id == "doc-1"
    assert len(ext) == 1
    assert ext[0].source_document_id == "doc-1"
    assert g.node_count == 0


def test_load_checkpoint_invalid_json_returns_none() -> None:
    store = {
        "documents.json": b"not json",
        "extraction_results.json": b"[]",
        "graph.json": b'{"nodes":[],"edges":[]}',
    }

    def get_file(key: str) -> bytes | None:
        return store.get(key)

    assert load_checkpoint_from_blobs(get_file, resume=True) is None
