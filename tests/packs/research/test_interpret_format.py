"""Research interpretation template formatting."""

from src.core.types import EvidenceItem, PatternCandidate, PatternType
from src.packs.research import ResearchPack
from src.packs.research.interpret_format import format_research_interpretation


def test_bridge_substitutes_node_names() -> None:
    pack = ResearchPack()
    p = PatternCandidate(
        pattern_type=PatternType.BRIDGE,
        title="Bridge title",
        details={
            "bridge_node_u": "Alpha",
            "bridge_node_v": "Beta",
            "community_u": 0,
            "community_v": 1,
        },
    )
    out = pack.interpret(p)
    assert "Alpha" in out
    assert "Beta" in out
    assert "{community_a}" not in out


def test_contradiction_uses_evidence_urls() -> None:
    pack = ResearchPack()
    p = PatternCandidate(
        pattern_type=PatternType.CONTRADICTION,
        title="C",
        evidence=[
            EvidenceItem(
                assertion_node_id="a1",
                assertion_text="Claim one",
                source_document_id="d1",
                source_url="https://a.example",
                source_tier=1,
            )
        ],
        counter_evidence=[
            EvidenceItem(
                assertion_node_id="a2",
                assertion_text="Claim two",
                source_document_id="d2",
                source_url="https://b.example",
                source_tier=2,
            )
        ],
    )
    out = pack.interpret(p)
    assert "Claim one" in out
    assert "https://a.example" in out
    assert "https://b.example" in out


def test_unknown_placeholder_preserved() -> None:
    out = format_research_interpretation(
        PatternCandidate(pattern_type=PatternType.BRIDGE, title="T"),
        "Hello {not_in_context} world",
    )
    assert "{not_in_context}" in out
    assert "Hello " in out
