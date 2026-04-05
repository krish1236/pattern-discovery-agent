"""Integration checks: graph round-trip, pack load, and mocked agent run."""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from unittest.mock import AsyncMock, patch

import sys
from types import ModuleType

import numpy as np
import pytest

try:
    import agent_runtime  # noqa: F401
except ModuleNotFoundError:
    _ar = ModuleType("agent_runtime")

    class _StubAgentRuntime:
        def agent(self, name: str = "", planned_steps=None):  # noqa: ANN001
            def _decorator(fn):
                return fn

            return _decorator

    _ar.AgentRuntime = _StubAgentRuntime
    sys.modules["agent_runtime"] = _ar

import agent as agent_module
from src.core.graph import KnowledgeGraph
from src.core.types import Edge, EdgeMeta, EdgeType, ExtractionResult, Node, NodeType, SourceDocument
from src.packs.research import ResearchPack

DOC_ID = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"


@pytest.mark.integration
def test_graph_roundtrip_and_pack_loads() -> None:
    g = KnowledgeGraph()
    meta = EdgeMeta(
        source_tier=1,
        source_family="scholarly",
        source_url="https://test.com",
        source_id="W1",
        extraction_confidence=0.9,
        extraction_model="test",
        timestamp="2025-01-01",
        provenance="abstract",
        domain="research",
    )
    n1 = Node(id="n1", node_type=NodeType.CONCEPT, name="Test A", domain="research")
    n2 = Node(id="n2", node_type=NodeType.ASSERTION, name="Test B", domain="research")
    g.add_node(n1)
    g.add_node(n2)
    g.add_edge(
        Edge(
            source_node_id="n1",
            target_node_id="n2",
            edge_type=EdgeType.ASSOCIATED_WITH,
            meta=meta,
        )
    )
    restored = KnowledgeGraph.from_json(g.to_json())
    assert restored.node_count == 2
    assert restored.edge_count == 1

    pack = ResearchPack()
    assert pack.domain == "research"
    assert "assertions" in pack.get_schema().extraction_prompt.lower()


def _fixture_doc() -> SourceDocument:
    return SourceDocument(
        id=DOC_ID,
        source_id="ext-1",
        source_family="scholarly",
        source_url="https://fixture.example/paper1",
        title="Fixture Paper",
        abstract="We recommend Quantum Method X for evaluation benchmarks.",
        source_tier=1,
        domain="research",
        precomputed_embedding=[0.02] * 16,
    )


def _edge_meta() -> EdgeMeta:
    return EdgeMeta(
        source_tier=1,
        source_family="scholarly",
        source_url="https://fixture.example/paper1",
        source_id="ext-1",
        extraction_confidence=0.9,
        extraction_model="test",
        timestamp="2020-01-01",
        provenance="abstract",
        domain="research",
    )


def make_gap_extraction_result(doc_id: str) -> ExtractionResult:
    m = _edge_meta()
    doc_node = Node(
        id=doc_id,
        node_type=NodeType.SOURCE_DOCUMENT,
        name="Fixture Paper",
        description="We recommend Quantum Method X",
        domain="research",
        properties={
            "source_url": "https://fixture.example/paper1",
            "publication_date": "2020-01-01",
            "source_tier": 1,
        },
        embedding=[0.02] * 16,
    )
    target = Node(
        id="concept-gap-node",
        node_type=NodeType.CONCEPT,
        name="Quantum Method X",
        description="Technique under discussion",
        domain="research",
    )
    a1 = Node(
        id="assert-g1",
        node_type=NodeType.ASSERTION,
        name="rec1",
        description="We strongly recommend Quantum Method X.",
        domain="research",
        properties={
            "source_document_id": doc_id,
            "source_url": "https://fixture.example/paper1",
            "source_tier": 1,
        },
    )
    a2 = Node(
        id="assert-g2",
        node_type=NodeType.ASSERTION,
        name="rec2",
        description="Future work should apply Quantum Method X.",
        domain="research",
        properties={
            "source_document_id": doc_id,
            "source_url": "https://fixture.example/paper2",
            "source_tier": 1,
        },
    )
    edges = [
        Edge(doc_id, a1.id, EdgeType.ASSERTS, m),
        Edge(doc_id, a2.id, EdgeType.ASSERTS, m),
        Edge(a1.id, target.id, EdgeType.RECOMMENDS, m),
        Edge(a2.id, target.id, EdgeType.RECOMMENDS, m),
    ]
    return ExtractionResult(
        source_document_id=doc_id,
        nodes=[doc_node, target, a1, a2],
        edges=edges,
    )


def _fake_embed_texts(texts: list[str], batch_size: int = 64) -> np.ndarray:
    h = hashlib.sha256((texts[0] if texts else "").encode()).hexdigest()
    seed = int(h[:8], 16)
    rng = np.random.default_rng(seed)
    n = len(texts)
    dim = 384
    v = rng.standard_normal((n, dim))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


class MockCtx:
    def __init__(self) -> None:
        self.state: dict = {}
        self.inputs: dict = {}
        self._blob_store: dict[str, bytes] = {}
        self.artifacts: list[tuple[str, str, str]] = []
        self.log_messages: list[tuple[str, str]] = []

        class Storage:
            def __init__(self, outer: MockCtx) -> None:
                self._o = outer

            def put_file(self, key: str, data: bytes, content_type: str = "application/json") -> None:
                self._o._blob_store[key] = bytes(data) if not isinstance(data, bytes) else data

            def get_file(self, key: str) -> bytes | None:
                return self._o._blob_store.get(key)

        self.storage = Storage(self)

        class Results:
            def __init__(self) -> None:
                self.stats: dict = {}
                self.tables: dict[str, list] = {}

            def set_stats(self, d: dict) -> None:
                self.stats = dict(d)

            def set_table(self, name: str, rows: list) -> None:
                self.tables[name] = list(rows)

        self.results = Results()

    @contextmanager
    def safe_step(self, name: str):
        yield

    def log(self, msg: str, level: str = "info") -> None:
        self.log_messages.append((level, str(msg)))

    def artifact(self, name: str, content: str, content_type: str) -> None:
        self.artifacts.append((name, content, content_type))


class StubPack(ResearchPack):
    def get_connectors(self, config):  # noqa: ANN001
        return [self._conn]


def _make_stub_pack() -> StubPack:
    p = StubPack()
    p._conn = _StubConnector()
    return p


class _StubConnector:
    def __init__(self) -> None:
        self.searches = 0

    async def search(self, query: str, limit: int = 20, **kwargs):
        self.searches += 1
        return [_fixture_doc()]

    async def get(self, source_id: str):
        return None

    async def expand(self, source_id: str, limit: int = 10):
        return []

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
@pytest.mark.integration
async def test_agent_pipeline_mocked_ingest_and_extract() -> None:
    ctx = MockCtx()
    ext_result = make_gap_extraction_result(DOC_ID)
    pack = _make_stub_pack()

    async def _extract_batch(docs, schema, api_key=None):  # noqa: ANN001
        assert len(docs) == 1
        assert docs[0].id == DOC_ID
        return [ext_result]

    with (
        patch.object(agent_module, "resolve_domain_pack", return_value=pack),
        patch.object(agent_module, "extract_batch", side_effect=_extract_batch),
        patch.object(agent_module, "expand_corpus", new_callable=AsyncMock, return_value=[]),
        patch.object(agent_module, "embed_texts", side_effect=_fake_embed_texts),
    ):
        out = await agent_module.run(
            ctx,
            {
                "topic": "quantum methods",
                "max_documents": 5,
                "focus": "gaps",
                "depth": "shallow",
            },
        )

    assert out["status"] == "completed"
    assert out["documents_analyzed"] == 1
    names = {a[0] for a in ctx.artifacts}
    assert "pattern_report.md" in names
    assert "evidence_table.json" in names
    assert "knowledge_graph.html" in names
    assert "run_summary.json" in names
    assert ctx.results.stats.get("documents_analyzed") == 1
    assert "patterns" in ctx.results.tables
    assert pack._conn.searches >= 1


@pytest.mark.asyncio
@pytest.mark.integration
async def test_agent_resume_skips_ingest_when_checkpoint_valid() -> None:
    doc = _fixture_doc()
    ext = make_gap_extraction_result(DOC_ID)
    graph = KnowledgeGraph()
    graph.add_extraction_result(ext)

    docs_blob = json.dumps([doc.to_dict()]).encode()
    ext_blob = json.dumps(
        [
            {
                "doc_id": ext.source_document_id,
                "nodes": [n.to_dict() for n in ext.nodes],
                "edges": [e.to_dict() for e in ext.edges],
            }
        ]
    ).encode()
    graph_blob = graph.to_json(include_embeddings=False).encode()

    ctx = MockCtx()
    ctx._blob_store["documents.json"] = docs_blob
    ctx._blob_store["extraction_results.json"] = ext_blob
    ctx._blob_store["graph.json"] = graph_blob

    pack = _make_stub_pack()

    with (
        patch.object(agent_module, "resolve_domain_pack", return_value=pack),
        patch.object(agent_module, "extract_batch", AsyncMock(side_effect=AssertionError("no extract on resume"))),
        patch.object(agent_module, "expand_corpus", AsyncMock(side_effect=AssertionError("no expand on resume"))),
        patch.object(agent_module, "embed_texts", side_effect=_fake_embed_texts),
    ):
        out = await agent_module.run(
            ctx,
            {"topic": "quantum", "max_documents": 5, "focus": "gaps", "resume": True},
        )

    assert out["status"] == "completed"
    assert pack._conn.searches == 0
