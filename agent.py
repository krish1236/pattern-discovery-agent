"""
Pattern Discovery Agent — RunForge entry point.

Orchestrates ingest, extraction, graph build, pattern mining, verification, and artifacts.
"""

from __future__ import annotations

import json
import logging
import os

from agent_runtime import AgentRuntime

from src.core.graph import KnowledgeGraph
from src.core.patterns import detect_bridges, detect_contradictions, detect_drift, detect_gaps
from src.core.report import (
    generate_coverage_markdown,
    generate_evidence_table,
    generate_graph_html,
    generate_pattern_report,
)
from src.checkpoint import load_checkpoint_from_blobs
from src.core.types import ConfidenceLevel, CoverageReport, ExtractionResult
from src.core.verifier import verify_all
from src.packs.research import ResearchPack
from src.packs.research.router import build_source_plan
from src.shared.corpus import assign_tiers, corpus_stats, deduplicate, expand_corpus
from src.shared.embeddings import embed_texts
from src.shared.extraction import extract_batch

logger = logging.getLogger(__name__)

runtime = AgentRuntime()

STEPS = [
    "classify_topic",
    "route_sources",
    "ingest_corpus",
    "expand_corpus",
    "extract_knowledge",
    "build_graph",
    "mine_patterns",
    "verify_patterns",
    "generate_report",
]


def _storage_put(ctx, key: str, text: str, content_type: str = "application/json") -> None:
    put = getattr(ctx.storage, "put_file", None)
    if not callable(put):
        return
    put(key, text.encode("utf-8"), content_type=content_type)


def _storage_get_file(ctx, key: str) -> bytes | None:
    get = getattr(ctx.storage, "get_file", None)
    if not callable(get):
        return None
    try:
        return get(key)
    except (AttributeError, OSError, TypeError):
        return None


def _embed_graph_text_nodes(ctx, graph: KnowledgeGraph) -> None:
    text_nodes = [
        n
        for n in graph._node_registry.values()
        if n.embedding is None and (n.description or n.name)
    ]
    if not text_nodes:
        return
    try:
        texts = [(n.description or n.name)[:8000] for n in text_nodes]
        embs = embed_texts(texts)
        for i, node in enumerate(text_nodes):
            if i < len(embs):
                row = embs[i]
                node.embedding = row.tolist() if hasattr(row, "tolist") else list(row)
    except Exception as e:
        ctx.log(f"Embedding pass failed: {e}", level="warning")


async def _close_connectors(connectors: list) -> None:
    for c in connectors:
        close = getattr(c, "close", None)
        if close and callable(close):
            try:
                await close()
            except Exception as e:
                logger.warning("Connector close failed: %s", e)


@runtime.agent(name="pattern-discovery", planned_steps=STEPS)
async def run(ctx, input):  # noqa: ANN001
    pack = ResearchPack()
    payload = {**(input or {}), **ctx.inputs}
    topic = payload.get("topic", "")
    depth = payload.get("depth", "standard")
    focus = payload.get("focus", "all")
    time_range = payload.get("time_range")
    max_documents = int(payload.get("max_documents", 100))

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    config = {
        "OPENALEX_API_KEY": os.environ.get("OPENALEX_API_KEY", ""),
        "SEMANTIC_SCHOLAR_API_KEY": os.environ.get("SEMANTIC_SCHOLAR_API_KEY", ""),
        "GITHUB_TOKEN": os.environ.get("GITHUB_TOKEN", ""),
        "TAVILY_API_KEY": os.environ.get("TAVILY_API_KEY", ""),
    }

    all_docs: list = []
    results: list[ExtractionResult] = []
    graph = KnowledgeGraph()
    promoted: list = []
    exploratory: list = []
    plan = build_source_plan(
        topic,
        depth=depth,
        focus=focus,
        time_range=time_range,
        max_documents=max_documents,
    )

    resume_requested = bool(payload.get("resume"))
    loaded = load_checkpoint_from_blobs(
        lambda k: _storage_get_file(ctx, k),
        resume=resume_requested,
    )
    skip_through_build = loaded is not None
    if loaded:
        all_docs, results, graph = loaded

    with ctx.safe_step("classify_topic"):
        ctx.log(f"Topic: {topic!r}, domain: {pack.domain}")
        ctx.state["topic"] = topic
        ctx.state["domain"] = pack.domain
        if resume_requested:
            if skip_through_build:
                ctx.log("Resume: loaded checkpoint blobs; ingest through graph build will be skipped.")
            else:
                ctx.log(
                    "Resume requested but checkpoint incomplete or invalid; running full pipeline.",
                    level="warning",
                )

    with ctx.safe_step("route_sources"):
        ctx.state["source_plan"] = {"connectors": plan.connectors, "queries": plan.queries}
        ctx.log(f"Source plan: {plan.connectors}, {len(plan.queries)} queries")

    with ctx.safe_step("ingest_corpus"):
        if skip_through_build:
            stats = corpus_stats(all_docs)
            ctx.state["corpus_stats"] = stats
            ctx.log(f"Resume: skipped ingest ({stats['total_documents']} documents from checkpoint)")
        else:
            connectors = pack.get_connectors(config)
            try:
                for connector in connectors:
                    for query in plan.queries[:3]:
                        try:
                            docs = await connector.search(
                                query,
                                limit=plan.per_connector_limit,
                                time_filter=plan.time_filter,
                            )
                            all_docs.extend(docs)
                            ctx.log(f"  {connector.__class__.__name__}: {len(docs)} docs for {query!r}")
                        except Exception as e:
                            ctx.log(f"  {connector.__class__.__name__} failed: {e}", level="warning")
            finally:
                await _close_connectors(connectors)

            all_docs = deduplicate(all_docs)
            all_docs = all_docs[:max_documents]
            all_docs = assign_tiers(all_docs, pack)
            stats = corpus_stats(all_docs)
            ctx.state["corpus_stats"] = stats
            ctx.log(f"Corpus: {stats['total_documents']} documents after dedup")
            _storage_put(ctx, "documents.json", json.dumps([d.to_dict() for d in all_docs]))

    with ctx.safe_step("expand_corpus"):
        if skip_through_build:
            ctx.log("Resume: skipped corpus expansion")
        else:
            connectors = pack.get_connectors(config)
            try:
                new_docs = await expand_corpus(all_docs, connectors, budget=plan.expansion_budget)
            finally:
                await _close_connectors(connectors)
            new_docs = assign_tiers(new_docs, pack)
            all_docs.extend(new_docs)
            all_docs = deduplicate(all_docs)
            ctx.log(f"After expansion: {len(all_docs)} documents")
            _storage_put(ctx, "documents.json", json.dumps([d.to_dict() for d in all_docs]))

    with ctx.safe_step("extract_knowledge"):
        if skip_through_build:
            ctx.state["extraction_stats"] = {
                "nodes": sum(len(r.nodes) for r in results),
                "edges": sum(len(r.edges) for r in results),
            }
            ctx.log(
                f"Resume: skipped extraction "
                f"({ctx.state['extraction_stats']['nodes']} nodes, "
                f"{ctx.state['extraction_stats']['edges']} edges from checkpoint)"
            )
        else:
            schema = pack.get_schema()
            if not api_key:
                ctx.log("ANTHROPIC_API_KEY missing; skipping LLM extraction.", level="warning")
                results = [ExtractionResult(source_document_id=d.id) for d in all_docs]
            else:
                try:
                    results = await extract_batch(all_docs, schema, api_key=api_key)
                except Exception as e:
                    ctx.log(f"Extraction failed: {e}", level="warning")
                    results = [ExtractionResult(source_document_id=d.id) for d in all_docs]

            ctx.state["extraction_stats"] = {
                "nodes": sum(len(r.nodes) for r in results),
                "edges": sum(len(r.edges) for r in results),
            }
            ctx.log(
                f"Extraction: {ctx.state['extraction_stats']['nodes']} nodes, "
                f"{ctx.state['extraction_stats']['edges']} edges"
            )
            _storage_put(
                ctx,
                "extraction_results.json",
                json.dumps(
                    [
                        {
                            "doc_id": r.source_document_id,
                            "nodes": [n.to_dict() for n in r.nodes],
                            "edges": [e.to_dict() for e in r.edges],
                        }
                        for r in results
                    ]
                ),
            )

    with ctx.safe_step("build_graph"):
        if skip_through_build:
            _embed_graph_text_nodes(ctx, graph)
            gstats = graph.stats()
            ctx.state["graph_stats"] = gstats
            ctx.log(
                f"Resume: graph from checkpoint — {gstats['total_nodes']} nodes, "
                f"{gstats['total_edges']} edges (embeddings refreshed if missing)"
            )
        else:
            graph = KnowledgeGraph()
            for result in results:
                graph.add_extraction_result(result)

            _embed_graph_text_nodes(ctx, graph)

            gstats = graph.stats()
            ctx.state["graph_stats"] = gstats
            ctx.log(f"Graph: {gstats['total_nodes']} nodes, {gstats['total_edges']} edges")
            _storage_put(ctx, "graph.json", graph.to_json())

    with ctx.safe_step("mine_patterns"):
        _embed_graph_text_nodes(ctx, graph)
        candidates = []
        if focus in ("bridges", "all"):
            candidates.extend(detect_bridges(graph))
        if focus in ("contradictions", "all"):
            try:
                candidates.extend(detect_contradictions(graph))
            except Exception as e:
                ctx.log(f"Contradiction mining failed: {e}", level="warning")
        if focus in ("drift", "all"):
            try:
                candidates.extend(detect_drift(graph))
            except Exception as e:
                ctx.log(f"Drift mining failed: {e}", level="warning")
        if focus in ("gaps", "all"):
            candidates.extend(detect_gaps(graph))

        ctx.state["pattern_stats"] = {"candidates": len(candidates)}
        ctx.log(f"Pattern mining: {len(candidates)} candidates")

    with ctx.safe_step("verify_patterns"):
        promoted, exploratory = verify_all(candidates)
        ctx.state["pattern_stats"]["promoted"] = len(promoted)
        ctx.state["pattern_stats"]["exploratory"] = len(exploratory)
        ctx.log(f"Verified: {len(promoted)} promoted, {len(exploratory)} exploratory")

    with ctx.safe_step("generate_report"):
        tier_map = corpus_stats(all_docs)["documents_per_tier"]
        coverage = CoverageReport(
            source_families_used=sorted({d.source_family for d in all_docs}),
            total_documents=len(all_docs),
            documents_per_tier=tier_map,
        )

        report_md = generate_pattern_report(promoted, exploratory, coverage)
        ctx.artifact("pattern_report.md", report_md, "text/markdown")

        evidence_json = generate_evidence_table(promoted, exploratory)
        ctx.artifact("evidence_table.json", evidence_json, "application/json")

        ctx.artifact("coverage_report.md", generate_coverage_markdown(coverage), "text/markdown")

        graph_html = generate_graph_html(graph, promoted)
        ctx.artifact("knowledge_graph.html", graph_html, "text/html")

        ctx.results.set_stats(
            {
                "patterns_found": len(promoted),
                "exploratory_leads": len(exploratory),
                "high_confidence": sum(
                    1 for p in promoted if p.confidence_level == ConfidenceLevel.HIGH
                ),
                "documents_analyzed": len(all_docs),
                "graph_nodes": graph.node_count,
                "graph_edges": graph.edge_count,
            }
        )
        ctx.results.set_table(
            "patterns",
            [
                {
                    "type": p.pattern_type.value,
                    "title": p.title,
                    "confidence": p.confidence_level.value if p.confidence_level else "",
                    "evidence_count": p.evidence_count,
                }
                for p in promoted
            ],
        )
        ctx.log(f"Artifacts written: {len(promoted)} promoted patterns, {len(all_docs)} documents")

    return {
        "status": "completed",
        "patterns_promoted": len(promoted),
        "patterns_exploratory": len(exploratory),
        "documents_analyzed": len(all_docs),
    }


if __name__ == "__main__":
    runtime.serve()
