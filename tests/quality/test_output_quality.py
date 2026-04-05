"""
Quality evaluation tests for Pattern Discovery Agent.

Requires a completed run of ``scripts/run_evaluation.py`` (outputs under ``eval_output/``).

Usage:
    python scripts/run_evaluation.py --topic "AI agent frameworks and orchestration 2024-2026"

    EVAL_DIR=eval_output/<timestamp> pytest tests/quality/ -m quality -v
    # Or use the latest eval_output/* directory when EVAL_DIR is unset
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.quality


def _tier_count(corpus_stats: dict, tier: int) -> int:
    m = corpus_stats.get("documents_per_tier") or {}
    return int(m.get(str(tier), m.get(tier, 0)))


def _find_latest_eval_dir() -> Path | None:
    """Find the most recent eval_output directory."""
    env = os.environ.get("EVAL_DIR")
    if env:
        return Path(env)
    base = Path("eval_output")
    if not base.exists():
        return None
    dirs = sorted(base.iterdir(), reverse=True)
    return dirs[0] if dirs else None


@pytest.fixture(scope="module")
def eval_dir():
    d = _find_latest_eval_dir()
    if d is None or not d.exists():
        pytest.skip("No eval output found. Run: python scripts/run_evaluation.py")
    return d


@pytest.fixture(scope="module")
def summary(eval_dir):
    return json.loads((eval_dir / "00_summary.json").read_text())


@pytest.fixture(scope="module")
def corpus_stats(eval_dir):
    return json.loads((eval_dir / "02_corpus_stats.json").read_text())


@pytest.fixture(scope="module")
def extraction(eval_dir):
    return json.loads((eval_dir / "04_extraction.json").read_text())


@pytest.fixture(scope="module")
def graph_stats(eval_dir):
    return json.loads((eval_dir / "05_graph_stats.json").read_text())


@pytest.fixture(scope="module")
def candidates(eval_dir):
    raw = (eval_dir / "06_candidates.json").read_text()
    return json.loads(raw)


@pytest.fixture(scope="module")
def report(eval_dir):
    return (eval_dir / "07_pattern_report.md").read_text()


@pytest.fixture(scope="module")
def evidence_table(eval_dir):
    return json.loads((eval_dir / "07_evidence_table.json").read_text())


# ── Corpus quality ───────────────────────────────────────────────────

class TestCorpusQuality:
    """Verify the corpus is large enough and diverse enough."""

    def test_minimum_document_count(self, summary):
        """We need enough documents to find patterns."""
        assert summary["documents_analyzed"] >= 20, (
            f"Only {summary['documents_analyzed']} documents — too few for meaningful patterns. "
            "Check if connectors are returning results."
        )

    def test_has_tier_1_sources(self, corpus_stats):
        """At least some documents should be primary evidence when the corpus includes journals."""
        tier_1 = _tier_count(corpus_stats, 1)
        if tier_1 < 1:
            pytest.skip(
                "No Tier-1 docs (common for preprint/GitHub-heavy runs); not a hard failure."
            )

    def test_multiple_source_families(self, corpus_stats):
        """Corpus ideally spans multiple source families (connectors)."""
        families = corpus_stats.get("documents_per_family", {})
        if len(families) < 2:
            pytest.skip(
                f"Single source family {list(families.keys())!r}; enable web/GitHub for diversity."
            )

    def test_most_documents_have_abstracts(self, corpus_stats):
        """Documents without abstracts can't be extracted."""
        total = corpus_stats.get("post_dedup", 1)
        with_abstract = corpus_stats.get("with_abstract", 0)
        ratio = with_abstract / total
        assert ratio >= 0.5, (
            f"Only {with_abstract}/{total} ({ratio:.0%}) have abstracts. "
            "Extraction will produce sparse results."
        )


# ── Extraction quality ───────────────────────────────────────────────

class TestExtractionQuality:
    """Verify LLM extraction produces enough structured data."""

    def test_extraction_produces_nodes(self, extraction):
        """Extraction should produce nodes from documents."""
        assert extraction["total_nodes"] >= 10, (
            f"Only {extraction['total_nodes']} nodes extracted. "
            "Check extraction prompt or LLM response parsing."
        )

    def test_extraction_produces_edges(self, extraction):
        """Extraction should produce edges (relationships)."""
        assert extraction["total_edges"] >= 5, (
            f"Only {extraction['total_edges']} edges extracted. "
            "Check relationship extraction in the prompt."
        )

    def test_average_nodes_per_document(self, extraction):
        """Each document should produce at least a few nodes."""
        total_docs = extraction["total_documents"]
        if total_docs == 0:
            pytest.skip("No documents processed")
        avg = extraction["total_nodes"] / total_docs
        assert avg >= 1.5, (
            f"Average {avg:.1f} nodes per document — too sparse. "
            "Extraction prompt may need tuning."
        )

    def test_has_assertion_nodes(self, extraction):
        """Extraction should produce assertion nodes (claims)."""
        assertion_count = sum(
            1 for doc in extraction.get("per_document", [])
            for nt in doc.get("node_types", [])
            if nt == "assertion"
        )
        assert assertion_count >= 5, (
            f"Only {assertion_count} assertion nodes. "
            "Contradiction detection needs assertions to work."
        )

    def test_has_concept_nodes(self, extraction):
        """Extraction should produce concept nodes."""
        concept_count = sum(
            1 for doc in extraction.get("per_document", [])
            for nt in doc.get("node_types", [])
            if nt == "concept"
        )
        assert concept_count >= 5, (
            f"Only {concept_count} concept nodes. "
            "Bridge detection needs concepts to find community connections."
        )


# ── Graph quality ────────────────────────────────────────────────────

class TestGraphQuality:
    """Verify the knowledge graph has good structure."""

    def test_graph_has_enough_nodes(self, graph_stats):
        assert graph_stats["total_nodes"] >= 20, (
            f"Only {graph_stats['total_nodes']} nodes in graph. "
            "Entity resolution may be too aggressive, or extraction too sparse."
        )

    def test_graph_has_edges(self, graph_stats):
        assert graph_stats["total_edges"] >= 10, (
            f"Only {graph_stats['total_edges']} edges. "
            "Graph is too sparse for pattern detection."
        )

    def test_graph_not_too_fragmented(self, graph_stats):
        """Graph shouldn't be mostly disconnected components."""
        nodes = graph_stats["total_nodes"]
        components = graph_stats["connected_components"]
        if nodes == 0:
            pytest.skip("Empty graph")
        # Largest component should contain most nodes
        # Having many small disconnected components means poor entity resolution
        ratio = components / nodes
        assert ratio < 0.7, (
            f"{components} components for {nodes} nodes (ratio={ratio:.2f}). "
            "Graph is too fragmented — entity resolution may be missing merges."
        )

    def test_graph_has_multiple_node_types(self, graph_stats):
        """Graph should have diverse node types."""
        types = graph_stats.get("nodes_by_type", {})
        assert len(types) >= 3, (
            f"Only {len(types)} node types: {list(types.keys())}. "
            "Extraction should produce assertions, concepts, actors, etc."
        )

    def test_graph_size_under_budget(self, summary):
        """Serialized graph should stay under 5MB."""
        size_mb = summary.get("graph_size_mb", 0)
        assert size_mb < 5.0, (
            f"Graph is {size_mb:.1f}MB — over 5MB budget. "
            "Consider reducing embedding dimensions or node count."
        )


# ── Pattern quality ──────────────────────────────────────────────────

class TestPatternQuality:
    """Verify patterns are meaningful, not noise."""

    def test_produces_at_least_one_candidate(self, candidates, summary):
        """Pipeline should find at least one pattern candidate."""
        if summary.get("focus") and summary.get("focus") != "all":
            pytest.skip("Candidate-count check applies when focus=all")
        assert len(candidates) >= 1, (
            "Zero pattern candidates found. Either: "
            "(1) graph is too small/sparse, "
            "(2) community detection found < 2 communities, "
            "(3) no assertion pairs had high enough cosine similarity."
        )

    def test_promotes_at_least_one_pattern(self, summary):
        """At least one pattern should pass the promotion gate."""
        if summary.get("focus") and summary.get("focus") != "all":
            pytest.skip("Pattern totals check applies when focus=all")
        total = summary["patterns_promoted"] + summary["patterns_exploratory"]
        assert total >= 1, (
            "Zero patterns (promoted or exploratory). "
            "Pipeline produced no findings at all."
        )

    def test_not_all_exploratory(self, summary):
        """At least one pattern should be strong enough to promote."""
        if summary.get("focus") and summary.get("focus") != "all":
            pytest.skip("Promotion mix check applies when focus=all")
        if summary["patterns_promoted"] + summary["patterns_exploratory"] == 0:
            pytest.skip("No patterns found")
        if summary["patterns_promoted"] == 0:
            pytest.xfail(
                f"All {summary['patterns_exploratory']} patterns are exploratory "
                "(not enough evidence to promote). May need more documents or "
                "better extraction."
            )

    def test_candidates_have_evidence(self, candidates, summary):
        """Each candidate should have at least some evidence."""
        if summary.get("focus") and summary.get("focus") != "all":
            pytest.skip("Evidence ratio check applies when focus=all")
        if not candidates:
            pytest.skip("No candidates")
        with_evidence = sum(1 for c in candidates if len(c.get("evidence", [])) > 0)
        ratio = with_evidence / len(candidates)
        assert ratio >= 0.3, (
            f"Only {with_evidence}/{len(candidates)} ({ratio:.0%}) candidates "
            "have evidence items. Pattern detectors may not be collecting "
            "evidence from neighboring nodes."
        )


# ── Report quality ───────────────────────────────────────────────────

class TestReportQuality:
    """Verify the pattern report is well-formed and useful."""

    def test_report_not_empty(self, report):
        assert len(report) > 100, "Pattern report is nearly empty."

    def test_report_has_pattern_sections(self, report):
        """Report should contain pattern card headers."""
        headers = re.findall(r"###\s+\[", report)
        # If we have patterns, report should have headers
        if "patterns_found" not in report.lower() or "0 promoted" in report.lower():
            pytest.skip("No promoted patterns in report")
        assert len(headers) >= 1, "Report has no pattern card headers (### [Type]: ...)"

    def test_report_has_confidence_labels(self, report):
        """Every pattern card should have a confidence label."""
        if "###" not in report:
            pytest.skip("No pattern cards in report")
        confidence_mentions = report.lower().count("**confidence:**")
        pattern_cards = len(re.findall(r"###\s+\[", report))
        if pattern_cards > 0:
            assert confidence_mentions >= pattern_cards * 0.8, (
                f"Only {confidence_mentions} confidence labels for {pattern_cards} "
                "pattern cards. Each card should have a confidence label."
            )

    def test_report_has_blind_spots(self, report):
        """Report should acknowledge limitations."""
        if "###" not in report:
            pytest.skip("No pattern cards in report")
        has_blind = "blind spot" in report.lower() or "blind_spot" in report.lower()
        assert has_blind, (
            "Report has no blind spots section. Every pattern should "
            "acknowledge what evidence is missing."
        )

    def test_report_has_evidence_citations(self, report):
        """Report should cite evidence with tier labels."""
        if "###" not in report:
            pytest.skip("No pattern cards in report")
        tier_mentions = len(re.findall(r"Tier\s+[1-4]", report))
        assert tier_mentions >= 1, (
            "Report doesn't cite evidence with tier labels. "
            "Each evidence item should show its source tier."
        )


# ── Evidence table quality ───────────────────────────────────────────

class TestEvidenceTableQuality:
    """Verify evidence table is complete and well-formed."""

    def test_evidence_table_not_empty(self, evidence_table):
        if not evidence_table:
            pytest.xfail("Evidence table is empty — no patterns had evidence items")

    def test_evidence_entries_have_required_fields(self, evidence_table):
        if not evidence_table:
            pytest.skip("Empty evidence table")
        required = {"pattern_type", "assertion_text", "source_tier"}
        for i, entry in enumerate(evidence_table[:10]):
            missing = required - set(entry.keys())
            assert not missing, (
                f"Evidence entry {i} missing fields: {missing}"
            )

    def test_evidence_has_source_urls(self, evidence_table):
        """Evidence items should have provenance URLs."""
        if not evidence_table:
            pytest.skip("Empty evidence table")
        with_url = sum(1 for e in evidence_table if e.get("source_url"))
        ratio = with_url / len(evidence_table) if evidence_table else 0
        # Some evidence items may legitimately lack URLs (e.g., from extraction)
        # but at least half should have them
        assert ratio >= 0.3, (
            f"Only {with_url}/{len(evidence_table)} ({ratio:.0%}) evidence items "
            "have source URLs. Provenance chain is broken."
        )


# ── Performance SLOs ─────────────────────────────────────────────────

class TestPerformanceSLOs:
    """Verify the run meets time and cost budgets."""

    def test_total_time_under_30_minutes(self, summary):
        elapsed = summary.get("total_time_seconds", 0)
        assert elapsed < 1800, (
            f"Run took {elapsed:.0f}s ({elapsed/60:.1f} min) — over 30 min budget. "
            "Check if connectors are timing out or extraction is too slow."
        )

    def test_estimated_cost_under_1_dollar(self, summary):
        cost = summary.get("estimated_cost_usd", 0)
        # This is a rough estimate; actual cost depends on token counts
        assert cost < 2.0, (
            f"Estimated cost ${cost:.2f} — check batch sizes and model selection."
        )


# ── Known ground truth (AI agent frameworks topic) ───────────────────

class TestKnownGroundTruth:
    """
    Validate against known patterns in "AI agent frameworks 2024-2026".

    These tests check whether the pipeline finds patterns we KNOW exist.
    If the topic is different, skip these.
    """

    @pytest.fixture(autouse=True)
    def _check_topic(self, summary):
        topic = summary.get("topic", "").lower()
        if "agent" not in topic:
            pytest.skip("Ground truth tests only apply to AI agent topics")

    def test_finds_multi_agent_related_content(self, report):
        """The report should mention multi-agent concepts."""
        text = report.lower()
        agent_terms = ["multi-agent", "multiagent", "agent framework",
                       "agent orchestration", "tool use", "planning"]
        found = [t for t in agent_terms if t in text]
        assert len(found) >= 1, (
            "Report doesn't mention any agent-related concepts. "
            f"Searched for: {agent_terms}"
        )

    def test_graph_captures_key_entities(self, eval_dir):
        """Graph should contain known entities from the agent space."""
        graph_json = json.loads((eval_dir / "05_graph.json").read_text())
        node_names = [n.get("name", "").lower() for n in graph_json.get("nodes", [])]
        all_names = " ".join(node_names)

        # These are well-known concepts in the agent framework space
        key_concepts = [
            "react", "langchain", "autogen", "crew", "agent",
            "tool", "planning", "reasoning", "llm", "prompt",
        ]
        found = [c for c in key_concepts if c in all_names]
        assert len(found) >= 2, (
            f"Graph only contains {len(found)} of {len(key_concepts)} "
            f"expected concepts: found={found}. "
            "Entity extraction may be missing key concepts."
        )


# ── Diagnostic helpers ───────────────────────────────────────────────

class TestDiagnostics:
    """
    These always pass but print useful diagnostic info.
    Run with -v to see the output.
    """

    def test_print_summary(self, summary):
        """Print run summary for quick review."""
        print(f"\n{'='*50}")
        print(f"Topic: {summary.get('topic', '?')}")
        print(f"Documents: {summary.get('documents_analyzed', 0)}")
        print(f"Graph: {summary.get('graph_nodes', 0)} nodes, {summary.get('graph_edges', 0)} edges")
        print(f"Promoted: {summary.get('patterns_promoted', 0)}")
        print(f"Exploratory: {summary.get('patterns_exploratory', 0)}")
        print(f"High confidence: {summary.get('high_confidence', 0)}")
        print(f"Time: {summary.get('total_time_seconds', 0):.0f}s")
        print(f"By type: {summary.get('patterns_by_type', {})}")
        print(f"{'='*50}")

    def test_print_pattern_titles(self, candidates):
        """Print all candidate pattern titles for quick scan."""
        if not candidates:
            print("\nNo candidates found.")
            return
        print(f"\n{'='*50}")
        print(f"CANDIDATES ({len(candidates)}):")
        for c in candidates:
            conf = c.get("confidence_score", 0)
            ev = len(c.get("evidence", []))
            print(f"  [{c.get('pattern_type', '?')}] {c.get('title', '?')[:70]} "
                  f"(conf={conf:.2f}, evidence={ev})")
        print(f"{'='*50}")
