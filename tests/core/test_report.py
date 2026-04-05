"""Report generation smoke tests."""

import json

from src.core.report import (
    generate_coverage_markdown,
    generate_evidence_table,
    generate_pattern_report,
    generate_patterns_summary,
    generate_run_summary,
)
from src.core.types import (
    ConfidenceLevel,
    CoverageReport,
    EvidenceItem,
    PatternType,
    PromotedPattern,
)


def test_generate_pattern_report_empty() -> None:
    cov = CoverageReport(source_families_used=["scholarly"], total_documents=0)
    md = generate_pattern_report([], [], cov)
    assert "Pattern Discovery Report" in md
    assert "Coverage" in md


def test_generate_evidence_table() -> None:
    p = PromotedPattern(
        pattern_type=PatternType.GAP,
        title="T",
        evidence=[
            EvidenceItem("a1", "text", "d1", "https://x.com", 1),
        ],
        confidence_score=0.8,
        interpretation="Gap guidance text",
    )
    p.confidence_level = ConfidenceLevel.MEDIUM
    ex = PromotedPattern(
        pattern_type=PatternType.BRIDGE,
        title="Weak",
        evidence=[
            EvidenceItem("a2", "e2", "d2", "https://y.com", 2),
        ],
        withheld_reason="Not enough evidence",
    )
    raw = generate_evidence_table([p], [ex])
    assert "pattern_id" in raw
    assert "https://x.com" in raw
    assert "promotion_status" in raw
    assert '"promoted"' in raw
    assert '"exploratory"' in raw
    assert "Gap guidance text" in raw
    assert "Not enough evidence" in raw


def test_generate_patterns_summary() -> None:
    pr = PromotedPattern(
        pattern_type=PatternType.BRIDGE,
        title="OK",
        evidence=[EvidenceItem("a1", "t", "d", "https://a", 1)] * 4,
        confidence_score=0.9,
        promotion_reason="Passed",
        interpretation="Interp",
    )
    pr.confidence_level = ConfidenceLevel.HIGH
    ex = PromotedPattern(
        pattern_type=PatternType.GAP,
        title="Weak",
        evidence=[EvidenceItem("a2", "t2", "d2", "https://b", 2)],
        withheld_reason="Low evidence",
    )
    raw = generate_patterns_summary([pr], [ex])
    assert "promoted" in raw
    assert "exploratory" in raw
    assert "Passed" in raw
    assert "Low evidence" in raw
    assert "counter_evidence_count" in raw


def test_generate_run_summary_roundtrip() -> None:
    snap = {
        "input": {"topic": "q", "resume": False},
        "patterns": {"promoted": 2, "exploratory": 1},
    }
    out = generate_run_summary(snap)
    assert json.loads(out) == snap


def test_coverage_markdown() -> None:
    md = generate_coverage_markdown(
        CoverageReport(source_families_used=["web"], total_documents=3, documents_per_tier={1: 2})
    )
    assert "Coverage report" in md
    assert "3" in md
