#!/usr/bin/env python3
"""
Run the Pattern Discovery Agent on a real topic and save all outputs
for manual quality review.

Usage:
    # Keys: set in environment or in project-root .env (loaded automatically)
    python scripts/run_evaluation.py

    python scripts/run_evaluation.py --topic "mRNA vaccine delivery mechanisms"
    python scripts/run_evaluation.py --depth quick

Outputs: eval_output/<timestamp>/

Then: pytest tests/quality/ -m quality -v
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Project root (parent of scripts/)
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.core.graph import KnowledgeGraph
from src.core.patterns import detect_bridges, detect_drift, detect_gaps
from src.core.patterns.dedup import dedup_promoted_contradictions
from src.core.patterns.contradictions import detect_contradictions, refine_contradiction_candidates
from src.core.report import (
    generate_coverage_markdown,
    generate_evidence_table,
    generate_graph_html,
    generate_pattern_report,
)
from src.core.types import (
    ConfidenceLevel,
    CoverageReport,
    PatternCandidate,
)
from src.core.verifier import verify_all
from src.packs.research import ResearchPack
from src.packs.research.router import build_source_plan
from src.shared.corpus import (
    assign_tiers,
    cap_documents_round_robin_by_family,
    corpus_stats,
    deduplicate,
    filter_by_relevance,
    expand_corpus,
)
from src.shared.embeddings import embed_texts
from src.shared.extraction import extract_batch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("eval")


def load_dotenv_file(path: Path | None = None) -> None:
    """Populate os.environ from .env if present (no extra dependencies)."""
    env_path = path or (_ROOT / ".env")
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        if not key:
            continue
        val = val.strip().strip("'").strip('"')
        # Do not overwrite explicit exports
        if key not in os.environ:
            os.environ[key] = val


def check_api_keys() -> dict[str, str]:
    """Check which API keys are available."""
    keys: dict[str, str] = {}
    required = {
        "ANTHROPIC_API_KEY": "LLM extraction + contradiction refine (REQUIRED)",
    }
    optional = {
        "OPENALEX_API_KEY": "OpenAlex scholarly search",
        "OPENALEX_MAILTO": "OpenAlex User-Agent mailto (polite pool)",
        "OPENALEX_EMAIL": "Alias for OPENALEX_MAILTO",
        "SEMANTIC_SCHOLAR_API_KEY": "Semantic Scholar (rate limits)",
        "TAVILY_API_KEY": "Web search for blogs/docs",
        "GITHUB_TOKEN": "GitHub repo search",
    }

    missing_required: list[str] = []
    for key, desc in required.items():
        val = os.environ.get(key, "")
        if val:
            keys[key] = val
            logger.info("  ✓ %s: set", key)
        else:
            missing_required.append(f"{key} — {desc}")
            logger.error("  ✗ %s: MISSING — %s", key, desc)

    for key, desc in optional.items():
        val = os.environ.get(key, "")
        if val:
            keys[key] = val
            logger.info("  ✓ %s: set", key)
        else:
            logger.warning("  ○ %s: not set — %s", key, desc)

    if missing_required:
        logger.error("\nMissing required keys: %s", ", ".join(missing_required))
        sys.exit(1)

    return keys


async def run_pipeline(
    topic: str,
    depth: str = "standard",
    focus: str = "all",
    time_range: str | None = None,
    max_documents: int = 100,
    output_dir: Path | None = None,
) -> dict:
    """Run the full pipeline and save all intermediate + final outputs."""

    if output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("eval_output") / ts
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Output directory: %s", output_dir)
    logger.info("Topic: %s", topic)
    logger.info("Depth: %s, Focus: %s, Max docs: %s", depth, focus, max_documents)

    keys = check_api_keys()
    pack = ResearchPack()
    api_key = keys["ANTHROPIC_API_KEY"]
    oa_mail = keys.get("OPENALEX_MAILTO") or keys.get("OPENALEX_EMAIL", "")
    config = {
        "OPENALEX_API_KEY": keys.get("OPENALEX_API_KEY", ""),
        "OPENALEX_MAILTO": str(oa_mail or ""),
        "OPENALEX_EMAIL": keys.get("OPENALEX_EMAIL", ""),
        "SEMANTIC_SCHOLAR_API_KEY": keys.get("SEMANTIC_SCHOLAR_API_KEY", ""),
        "GITHUB_TOKEN": keys.get("GITHUB_TOKEN", ""),
        "TAVILY_API_KEY": keys.get("TAVILY_API_KEY", ""),
    }

    timings: dict[str, float] = {}
    costs: dict[str, float] = {}

    # ── Step 1-2: Route ──────────────────────────────────────────────
    t0 = time.time()
    plan = build_source_plan(
        topic,
        depth=depth,
        focus=focus,
        time_range=time_range,
        max_documents=max_documents,
    )
    logger.info("Source plan: %s, %s queries", plan.connectors, len(plan.queries))

    (output_dir / "01_source_plan.json").write_text(
        json.dumps(
            {
                "topic": topic,
                "depth": depth,
                "focus": focus,
                "connectors": plan.connectors,
                "queries": plan.queries,
                "max_documents": plan.max_documents,
                "expansion_budget": plan.expansion_budget,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # ── Step 3: Ingest ───────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 3: Ingesting corpus")
    logger.info("=" * 60)
    t_ingest = time.time()

    connectors = pack.get_connectors(config)
    all_docs: list = []
    connector_stats: dict[str, int] = {}

    for connector in connectors:
        cname = connector.__class__.__name__
        cdocs: list = []
        for query in plan.queries[:8]:
            try:
                docs = await connector.search(
                    query,
                    limit=plan.per_connector_limit,
                    time_filter=plan.time_filter,
                )
                cdocs.extend(docs)
                logger.info("  %s: %s docs for %r", cname, len(docs), query)
            except Exception as e:
                logger.warning("  %s failed on %r: %s", cname, query, e)
        all_docs.extend(cdocs)
        connector_stats[cname] = len(cdocs)
        try:
            await connector.close()
        except Exception:
            pass

    pre_dedup = len(all_docs)
    all_docs = deduplicate(all_docs)
    all_docs = filter_by_relevance(all_docs, topic)
    all_docs = cap_documents_round_robin_by_family(all_docs, max_documents)
    all_docs = assign_tiers(all_docs, pack)
    stats = corpus_stats(all_docs)

    timings["ingest"] = time.time() - t_ingest
    logger.info("Corpus: %s raw → %s after dedup", pre_dedup, stats["total_documents"])
    logger.info("  Per tier: %s", stats["documents_per_tier"])
    logger.info("  Per family: %s", stats["documents_per_family"])
    logger.info("  With abstracts: %s", stats["with_abstract"])

    (output_dir / "02_corpus.json").write_text(
        json.dumps([d.to_dict() for d in all_docs], indent=2),
        encoding="utf-8",
    )
    (output_dir / "02_corpus_stats.json").write_text(
        json.dumps(
            {
                "pre_dedup": pre_dedup,
                "post_dedup": stats["total_documents"],
                "connector_stats": connector_stats,
                **stats,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # ── Step 4: Expand ───────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 4: Expanding corpus (1-hop citations)")
    logger.info("=" * 60)
    t_expand = time.time()

    connectors = pack.get_connectors(config)
    try:
        new_docs = await expand_corpus(all_docs, connectors, budget=plan.expansion_budget)
    finally:
        for c in connectors:
            try:
                await c.close()
            except Exception:
                pass
    new_docs = assign_tiers(new_docs, pack)
    all_docs.extend(new_docs)
    all_docs = deduplicate(all_docs)
    all_docs = filter_by_relevance(all_docs, topic)

    timings["expand"] = time.time() - t_expand
    logger.info("After expansion: %s documents (+%s new)", len(all_docs), len(new_docs))

    (output_dir / "03_corpus_expanded.json").write_text(
        json.dumps([d.to_dict() for d in all_docs], indent=2),
        encoding="utf-8",
    )

    # ── Step 5: Extract ──────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 5: LLM extraction (Haiku)")
    logger.info("=" * 60)
    t_extract = time.time()

    schema = pack.get_schema()
    results = await extract_batch(all_docs, schema, api_key=api_key)

    total_nodes = sum(len(r.nodes) for r in results)
    total_edges = sum(len(r.edges) for r in results)
    timings["extract"] = time.time() - t_extract

    avg_abstract_tokens = 200
    total_input_tokens = len(all_docs) * avg_abstract_tokens
    total_output_tokens = total_nodes * 50
    costs["extraction"] = (total_input_tokens * 0.25 + total_output_tokens * 1.25) / 1_000_000

    logger.info("Extracted: %s nodes, %s edges", total_nodes, total_edges)
    logger.info("  Time: %.1fs", timings["extract"])
    logger.info("  Est. cost: $%.3f", costs["extraction"])

    (output_dir / "04_extraction.json").write_text(
        json.dumps(
            {
                "total_documents": len(results),
                "total_nodes": total_nodes,
                "total_edges": total_edges,
                "per_document": [
                    {
                        "doc_id": r.source_document_id,
                        "nodes": len(r.nodes),
                        "edges": len(r.edges),
                        "node_types": [n.node_type.value for n in r.nodes],
                    }
                    for r in results
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # ── Step 6: Build graph ──────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 6: Building knowledge graph")
    logger.info("=" * 60)
    t_graph = time.time()

    graph = KnowledgeGraph()
    for result in results:
        graph.add_extraction_result(result)

    text_nodes = [
        n
        for n in graph._node_registry.values()
        if n.embedding is None and (n.description or n.name)
    ]
    if text_nodes:
        texts = [(n.description or n.name)[:8000] for n in text_nodes]
        embs = embed_texts(texts)
        for i, node in enumerate(text_nodes):
            if i < len(embs):
                row = embs[i]
                node.embedding = row.tolist() if hasattr(row, "tolist") else list(row)

    n_merged = graph.merge_nodes_by_embedding(min_cosine=0.85)
    if n_merged:
        logger.info("  Embedding merge: collapsed %s entity pairs", n_merged)

    gstats = graph.stats()
    timings["graph"] = time.time() - t_graph

    logger.info("Graph: %s nodes, %s edges", gstats["total_nodes"], gstats["total_edges"])
    logger.info("  Components: %s", gstats["connected_components"])
    logger.info("  Nodes by type: %s", gstats["nodes_by_type"])
    logger.info("  Edges by type: %s", gstats["edges_by_type"])

    graph_json = graph.to_json(include_embeddings=True)
    graph_size_mb = len(graph_json.encode()) / (1024 * 1024)
    logger.info("  Serialized size: %.1f MB", graph_size_mb)

    (output_dir / "05_graph.json").write_text(graph_json, encoding="utf-8")
    (output_dir / "05_graph_stats.json").write_text(json.dumps(gstats, indent=2), encoding="utf-8")

    # ── Step 7: Mine patterns ────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 7: Mining patterns")
    logger.info("=" * 60)
    t_mine = time.time()

    candidates: list[PatternCandidate] = []

    if focus in ("bridges", "all"):
        bridges = detect_bridges(graph)
        logger.info("  Bridges: %s candidates", len(bridges))
        candidates.extend(bridges)

    if focus in ("contradictions", "all"):
        try:
            contras = detect_contradictions(graph)
            logger.info("  Contradictions (NLI): %s candidates", len(contras))

            if contras and api_key:
                tpl = pack.get_schema().interpretation_templates.get("contradiction", "")
                if tpl:
                    pre_refine = len(contras)
                    contras = await refine_contradiction_candidates(contras, tpl, api_key)
                    logger.info(
                        "  Contradictions (after LLM refine): %s (dropped %s apparent)",
                        len(contras),
                        pre_refine - len(contras),
                    )
            candidates.extend(contras)
        except Exception as e:
            logger.warning("  Contradiction detection failed: %s", e)

    if focus in ("drift", "all"):
        try:
            drifts = detect_drift(graph)
            logger.info("  Drift: %s candidates", len(drifts))
            candidates.extend(drifts)
        except Exception as e:
            logger.warning("  Drift detection failed: %s", e)

    if focus in ("gaps", "all"):
        gaps = detect_gaps(graph)
        logger.info("  Gaps: %s candidates", len(gaps))
        candidates.extend(gaps)

    timings["mine"] = time.time() - t_mine
    logger.info("Total candidates: %s", len(candidates))

    (output_dir / "06_candidates.json").write_text(
        json.dumps([c.to_dict() for c in candidates], indent=2),
        encoding="utf-8",
    )

    # ── Step 8: Verify ───────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 8: Verifying and promoting patterns")
    logger.info("=" * 60)
    t_verify = time.time()

    promoted, exploratory = verify_all(candidates, interpret=pack.interpret)
    timings["verify"] = time.time() - t_verify

    promoted, dedup_stats = dedup_promoted_contradictions(promoted)
    if dedup_stats["patterns_merged"] > 0:
        logger.info(
            "Contradiction dedup: %s → %s contradictions (%s groups merged, %s patterns folded)",
            dedup_stats["contradictions_before"],
            dedup_stats["contradictions_after"],
            dedup_stats["groups_merged"],
            dedup_stats["patterns_merged"],
        )

    logger.info("Promoted: %s", len(promoted))
    for p in promoted:
        logger.info(
            "  [%s] %s (confidence=%s, evidence=%s)",
            p.pattern_type.value,
            p.title,
            p.confidence_level.value if p.confidence_level else "?",
            p.evidence_count,
        )
    logger.info("Exploratory: %s", len(exploratory))
    for e in exploratory:
        logger.info("  [%s] %s (withheld: %s)", e.pattern_type.value, e.title, e.withheld_reason)

    # ── Step 9: Report ───────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 9: Generating reports")
    logger.info("=" * 60)

    tier_map = corpus_stats(all_docs)["documents_per_tier"]
    coverage = CoverageReport(
        source_families_used=sorted({d.source_family for d in all_docs}),
        total_documents=len(all_docs),
        documents_per_tier=tier_map,
    )

    report_md = generate_pattern_report(promoted, exploratory, coverage)
    evidence_json = generate_evidence_table(promoted, exploratory)
    coverage_md = generate_coverage_markdown(coverage)
    graph_html = generate_graph_html(graph, promoted, exploratory)

    (output_dir / "07_pattern_report.md").write_text(report_md, encoding="utf-8")
    (output_dir / "07_evidence_table.json").write_text(evidence_json, encoding="utf-8")
    (output_dir / "07_coverage_report.md").write_text(coverage_md, encoding="utf-8")
    (output_dir / "07_knowledge_graph.html").write_text(graph_html, encoding="utf-8")

    total_time = time.time() - t0
    total_cost = sum(costs.values())

    summary = {
        "topic": topic,
        "depth": depth,
        "focus": focus,
        "documents_analyzed": len(all_docs),
        "graph_nodes": gstats["total_nodes"],
        "graph_edges": gstats["total_edges"],
        "graph_size_mb": round(graph_size_mb, 2),
        "patterns_promoted": len(promoted),
        "patterns_exploratory": len(exploratory),
        "patterns_by_type": {},
        "high_confidence": sum(
            1 for p in promoted if p.confidence_level == ConfidenceLevel.HIGH
        ),
        "timings_seconds": {k: round(v, 1) for k, v in timings.items()},
        "total_time_seconds": round(total_time, 1),
        "estimated_cost_usd": round(total_cost, 4),
        "output_dir": str(output_dir),
    }

    for p in promoted + exploratory:
        pt = p.pattern_type.value
        summary["patterns_by_type"][pt] = summary["patterns_by_type"].get(pt, 0) + 1

    (output_dir / "00_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info("=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 60)
    logger.info("Total time: %.0fs (%.1f min)", total_time, total_time / 60)
    logger.info("Estimated LLM cost: $%.4f", total_cost)
    logger.info("Documents: %s", len(all_docs))
    logger.info("Graph: %s nodes, %s edges", gstats["total_nodes"], gstats["total_edges"])
    logger.info("Promoted: %s, Exploratory: %s", len(promoted), len(exploratory))
    logger.info("\nAll outputs saved to: %s/", output_dir)
    logger.info("\nKey files to review:")
    logger.info("  %s/07_pattern_report.md     ← main findings", output_dir)
    logger.info("  %s/07_knowledge_graph.html  ← interactive graph", output_dir)
    logger.info("  %s/06_candidates.json       ← raw candidates", output_dir)
    logger.info("  %s/00_summary.json          ← run stats", output_dir)

    return summary


def main() -> None:
    load_dotenv_file()
    parser = argparse.ArgumentParser(description="Run Pattern Discovery evaluation")
    parser.add_argument(
        "--topic",
        default="AI agent frameworks and orchestration 2024-2026",
        help="Topic to analyze",
    )
    parser.add_argument(
        "--depth",
        choices=["quick", "standard", "deep"],
        default="standard",
        help="Search depth (quick=50, standard=100, deep=200 docs)",
    )
    parser.add_argument(
        "--focus",
        choices=["bridges", "contradictions", "drift", "gaps", "all"],
        default="all",
        help="Which pattern types to detect",
    )
    parser.add_argument(
        "--time-range",
        default=None,
        help="Time range filter, e.g. '2024-2026'",
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        default=100,
        help="Maximum documents to analyze",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: eval_output/<timestamp>)",
    )
    args = parser.parse_args()

    out = Path(args.output_dir) if args.output_dir else None
    asyncio.run(
        run_pipeline(
            topic=args.topic,
            depth=args.depth,
            focus=args.focus,
            time_range=args.time_range,
            max_documents=args.max_documents,
            output_dir=out,
        )
    )

    print("\n" + "=" * 60)
    print("QUALITY REVIEW CHECKLIST")
    print("=" * 60)
    print(
        """
Open 07_pattern_report.md and check each pattern.

Then run: pytest tests/quality/ -m quality -v
"""
    )


if __name__ == "__main__":
    main()
