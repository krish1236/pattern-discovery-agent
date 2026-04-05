"""Report generation smoke tests."""

from src.core.report import (
    generate_coverage_markdown,
    generate_evidence_table,
    generate_pattern_report,
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
    )
    p.confidence_level = ConfidenceLevel.MEDIUM
    raw = generate_evidence_table([p], [])
    assert "pattern_id" in raw
    assert "https://x.com" in raw


def test_coverage_markdown() -> None:
    md = generate_coverage_markdown(
        CoverageReport(source_families_used=["web"], total_documents=3, documents_per_tier={1: 2})
    )
    assert "Coverage report" in md
    assert "3" in md
