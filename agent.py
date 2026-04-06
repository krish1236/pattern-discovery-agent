"""
Pattern Discovery Agent — RunForge entry point.

Orchestrates ingest, extraction, graph build, pattern mining, verification, and artifacts.

Payload flags (merged with ``ctx.inputs``) include ``resume``, ``remine_patterns``, ``domain``,
``focus``, ``depth``, ``topic``, and ``max_documents``; see implementation in ``run`` below.
"""

from __future__ import annotations

import json
import logging
import os

from agent_runtime import AgentRuntime

from src.core.graph import KnowledgeGraph
from src.core.patterns import detect_bridges, detect_drift, detect_gaps
from src.core.patterns.contradictions import detect_contradictions, refine_contradiction_candidates
from src.core.report import (
    generate_coverage_markdown,
    generate_evidence_table,
    generate_graph_html,
    generate_pattern_report,
    generate_patterns_summary,
    generate_run_summary,
)
from src.checkpoint import load_checkpoint_from_blobs
from src.core.types import (
    ConfidenceLevel,
    CoverageReport,
    ExtractionResult,
    PatternCandidate,
    PromotedPattern,
)
from src.core.verifier import verify_all
from src.pack_registry import resolve_domain_pack
from src.packs.research.router import build_source_plan
from src.shared.corpus import (
    assign_tiers,
    cap_documents_round_robin_by_family,
    corpus_stats,
    deduplicate,
    expand_corpus,
    filter_by_relevance,
)
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


def _record_pipeline_error(ctx, step: str, message: str) -> None:
    """Append a non-fatal error for run_summary / logs; does not stop the run."""
    msg = str(message)[:4000]
    ctx.state.setdefault("pipeline_errors", []).append({"step": step, "message": msg})
    ctx.log(f"Non-fatal [{step}]: {msg}", level="warning")


def _safe_filter_by_relevance(ctx, step: str, documents: list, topic: str) -> list:
    if not documents or not (topic or "").strip():
        return documents
    try:
        return filter_by_relevance(documents, topic)
    except Exception as e:
        _record_pipeline_error(ctx, step, f"filter_by_relevance: {e}")
        return documents


@runtime.agent(name="pattern-discovery", planned_steps=STEPS)
async def run(ctx, input):  # noqa: ANN001
    ctx.state["pipeline_errors"] = []
    payload = {**(input or {}), **ctx.inputs}
    topic = payload.get("topic", "")
    depth = payload.get("depth", "standard")
    focus = payload.get("focus", "all")
    time_range = payload.get("time_range")
    max_documents = int(payload.get("max_documents", 100))

    pack = resolve_domain_pack(domain=payload.get("domain"), topic=topic)

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    config = {
        "OPENALEX_API_KEY": os.environ.get("OPENALEX_API_KEY", ""),
        "OPENALEX_MAILTO": os.environ.get("OPENALEX_MAILTO", ""),
        "OPENALEX_EMAIL": os.environ.get("OPENALEX_EMAIL", ""),
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
        ctx.log(f"Topic: {topic!r}, pack: {pack.domain}")
        ctx.state["topic"] = topic
        ctx.state["domain"] = pack.domain
        ctx.state["pack_domain"] = pack.domain
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
            connectors: list = []
            try:
                connectors = pack.get_connectors(config)
            except Exception as e:
                _record_pipeline_error(ctx, "ingest_corpus", f"get_connectors: {e}")
            try:
                for connector in connectors:
                    for query in plan.queries[:8]:
                        try:
                            docs = await connector.search(
                                query,
                                limit=plan.per_connector_limit,
                                time_filter=plan.time_filter,
                            )
                            all_docs.extend(docs)
                            ctx.log(f"  {connector.__class__.__name__}: {len(docs)} docs for {query!r}")
                        except Exception as e:
                            _record_pipeline_error(
                                ctx,
                                "ingest_corpus",
                                f"{connector.__class__.__name__}.search({query!r}): {e}",
                            )
            finally:
                await _close_connectors(connectors)

            try:
                all_docs = deduplicate(all_docs)
            except Exception as e:
                _record_pipeline_error(ctx, "ingest_corpus", f"deduplicate: {e}")
            all_docs = _safe_filter_by_relevance(ctx, "ingest_corpus", all_docs, topic)
            try:
                all_docs = cap_documents_round_robin_by_family(all_docs, max_documents)
            except Exception as e:
                _record_pipeline_error(ctx, "ingest_corpus", f"cap_documents: {e}")
            try:
                all_docs = assign_tiers(all_docs, pack)
            except Exception as e:
                _record_pipeline_error(ctx, "ingest_corpus", f"assign_tiers: {e}")
            stats = corpus_stats(all_docs)
            ctx.state["corpus_stats"] = stats
            ctx.log(f"Corpus: {stats['total_documents']} documents after dedup")
            _storage_put(ctx, "documents.json", json.dumps([d.to_dict() for d in all_docs]))

    with ctx.safe_step("expand_corpus"):
        if skip_through_build:
            ctx.log("Resume: skipped corpus expansion")
        elif len(all_docs) == 0:
            ctx.log("Skipped corpus expansion (empty corpus)")
        else:
            connectors = []
            try:
                connectors = pack.get_connectors(config)
            except Exception as e:
                _record_pipeline_error(ctx, "expand_corpus", f"get_connectors: {e}")
            new_docs: list = []
            try:
                new_docs = await expand_corpus(all_docs, connectors, budget=plan.expansion_budget)
            except Exception as e:
                _record_pipeline_error(ctx, "expand_corpus", f"expand_corpus: {e}")
            finally:
                await _close_connectors(connectors)
            try:
                new_docs = assign_tiers(new_docs, pack)
            except Exception as e:
                _record_pipeline_error(ctx, "expand_corpus", f"assign_tiers (expansion): {e}")
            all_docs.extend(new_docs)
            try:
                all_docs = deduplicate(all_docs)
            except Exception as e:
                _record_pipeline_error(ctx, "expand_corpus", f"deduplicate: {e}")
            all_docs = _safe_filter_by_relevance(ctx, "expand_corpus", all_docs, topic)
            ctx.log(f"After expansion: {len(all_docs)} documents")
            _storage_put(ctx, "documents.json", json.dumps([d.to_dict() for d in all_docs]))

    with ctx.safe_step("extract_knowledge"):
        empty_corpus = len(all_docs) == 0
        ctx.state["empty_corpus"] = empty_corpus
        if empty_corpus:
            ctx.log("Corpus is empty; extraction and graph will have no document content.", level="warning")

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
            if empty_corpus:
                results = []
                ctx.log("Skipped LLM extraction (empty corpus).")
            elif not api_key:
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
            graph.merge_nodes_by_embedding(min_cosine=0.85)
            gstats = graph.stats()
            ctx.state["graph_stats"] = gstats
            ctx.log(
                f"Resume: graph from checkpoint — {gstats['total_nodes']} nodes, "
                f"{gstats['total_edges']} edges (embeddings refreshed if missing)"
            )
        else:
            graph = KnowledgeGraph()
            for result in results:
                try:
                    graph.add_extraction_result(result)
                except Exception as e:
                    _record_pipeline_error(ctx, "build_graph", f"add_extraction_result: {e}")

            _embed_graph_text_nodes(ctx, graph)
            n_emb_merges = 0
            try:
                n_emb_merges = graph.merge_nodes_by_embedding(min_cosine=0.85)
            except Exception as e:
                _record_pipeline_error(ctx, "build_graph", f"merge_nodes_by_embedding: {e}")
            if n_emb_merges:
                ctx.log(f"Graph: merged {n_emb_merges} entity pairs by embedding similarity")

            gstats = graph.stats()
            ctx.state["graph_stats"] = gstats
            ctx.log(f"Graph: {gstats['total_nodes']} nodes, {gstats['total_edges']} edges")
            _storage_put(ctx, "graph.json", graph.to_json(include_embeddings=True))

    with ctx.safe_step("mine_patterns"):
        candidates: list[PatternCandidate] = []
        remine = bool(payload.get("remine_patterns"))
        if skip_through_build and not remine:
            raw_c = _storage_get_file(ctx, "pattern_candidates.json")
            if raw_c:
                try:
                    arr = json.loads(raw_c.decode("utf-8"))
                    candidates = [PatternCandidate.from_dict(x) for x in arr]
                    ctx.log(
                        f"Resume: using {len(candidates)} pattern candidates from storage "
                        "(set remine_patterns to re-run miners)"
                    )
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                    ctx.log(f"Could not load pattern_candidates.json: {e}", level="warning")

        if not candidates:
            _embed_graph_text_nodes(ctx, graph)
            graph.merge_nodes_by_embedding(min_cosine=0.85)
            if focus in ("bridges", "all"):
                try:
                    candidates.extend(detect_bridges(graph))
                except Exception as e:
                    _record_pipeline_error(ctx, "mine_patterns", f"detect_bridges: {e}")
            if focus in ("contradictions", "all"):
                try:
                    cd = detect_contradictions(graph)
                    tpl = pack.get_schema().interpretation_templates.get("contradiction", "")
                    if api_key and tpl:
                        try:
                            cd = await refine_contradiction_candidates(cd, tpl, api_key)
                        except Exception as e:
                            _record_pipeline_error(
                                ctx, "mine_patterns", f"refine_contradiction_candidates: {e}"
                            )
                    candidates.extend(cd)
                except Exception as e:
                    _record_pipeline_error(ctx, "mine_patterns", f"detect_contradictions: {e}")
            if focus in ("drift", "all"):
                try:
                    candidates.extend(detect_drift(graph))
                except Exception as e:
                    _record_pipeline_error(ctx, "mine_patterns", f"detect_drift: {e}")
            if focus in ("gaps", "all"):
                try:
                    candidates.extend(detect_gaps(graph))
                except Exception as e:
                    _record_pipeline_error(ctx, "mine_patterns", f"detect_gaps: {e}")

        ctx.state["pattern_stats"] = {"candidates": len(candidates)}
        ctx.log(f"Pattern mining: {len(candidates)} candidates")
        _storage_put(
            ctx,
            "pattern_candidates.json",
            json.dumps([c.to_dict() for c in candidates]),
        )

    with ctx.safe_step("verify_patterns"):
        try:
            promoted, exploratory = verify_all(candidates, interpret=pack.interpret)
        except Exception as e:
            _record_pipeline_error(ctx, "verify_patterns", str(e))
            promoted, exploratory = [], []
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

        ctx.artifact(
            "patterns_summary.json",
            generate_patterns_summary(promoted, exploratory),
            "application/json",
        )

        plan_info = ctx.state.get("source_plan") or {}
        run_snapshot: dict = {
            "empty_corpus": bool(ctx.state.get("empty_corpus")),
            "input": {
                "topic": topic,
                "depth": depth,
                "focus": focus,
                "time_range": time_range,
                "max_documents": max_documents,
                "resume": resume_requested,
                "remine_patterns": bool(payload.get("remine_patterns")),
                "domain": payload.get("domain"),
            },
            "resolved": {"pack_domain": pack.domain},
            "corpus": ctx.state.get("corpus_stats"),
            "extraction": ctx.state.get("extraction_stats"),
            "graph": ctx.state.get("graph_stats"),
            "patterns": ctx.state.get("pattern_stats"),
        }
        if isinstance(plan_info, dict):
            run_snapshot["source_plan"] = {
                "connectors": plan_info.get("connectors"),
                "query_count": len(plan_info.get("queries") or []),
            }
        errs = ctx.state.get("pipeline_errors") or []
        if errs:
            run_snapshot["pipeline_errors"] = errs
        ctx.artifact("run_summary.json", generate_run_summary(run_snapshot), "application/json")

        ctx.artifact("coverage_report.md", generate_coverage_markdown(coverage), "text/markdown")

        graph_html = generate_graph_html(graph, promoted, exploratory)
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
                "empty_corpus": bool(ctx.state.get("empty_corpus")),
                "pipeline_error_count": len(ctx.state.get("pipeline_errors") or []),
            }
        )
        def _pattern_result_row(p: PromotedPattern) -> dict[str, str | int]:
            return {
                "type": p.pattern_type.value,
                "title": p.title,
                "confidence": p.confidence_level.value if p.confidence_level else "",
                "evidence_count": p.evidence_count,
                "interpretation": p.interpretation or "",
                "withheld_reason": p.withheld_reason or "",
            }

        ctx.results.set_table("patterns", [_pattern_result_row(p) for p in promoted])
        ctx.results.set_table(
            "exploratory_patterns",
            [_pattern_result_row(p) for p in exploratory],
        )
        ctx.log(f"Artifacts written: {len(promoted)} promoted patterns, {len(all_docs)} documents")

    return {
        "status": "completed",
        "patterns_promoted": len(promoted),
        "patterns_exploratory": len(exploratory),
        "documents_analyzed": len(all_docs),
        "pipeline_errors": ctx.state.get("pipeline_errors", []),
    }


if __name__ == "__main__":
    runtime.serve()
