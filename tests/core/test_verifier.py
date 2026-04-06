"""Verifier and promotion gate tests."""

import pytest

from src.core.types import ConfidenceLevel, EvidenceItem, PatternCandidate, PatternType
from src.core.verifier import verify_all, verify_pattern


def _make_evidence(count: int, tiers: list[int] | None = None) -> list[EvidenceItem]:
    if tiers is None:
        tiers = [1] * count
    return [
        EvidenceItem(
            assertion_node_id=f"a{i}",
            assertion_text=f"text {i}",
            source_document_id=f"d{i}",
            source_url=f"https://source{i}.com",
            source_tier=tiers[i % len(tiers)],
        )
        for i in range(count)
    ]


class TestVerifier:
    def test_high_confidence_promotion(self) -> None:
        pattern = PatternCandidate(
            pattern_type=PatternType.BRIDGE,
            title="Strong bridge",
            evidence=_make_evidence(5, [1, 1, 2, 2, 1]),
            confidence_score=0.8,
        )
        result = verify_pattern(pattern)
        assert result is not None
        assert result.withheld_reason is None
        assert result.confidence_level == ConfidenceLevel.HIGH

    def test_medium_confidence(self) -> None:
        pattern = PatternCandidate(
            pattern_type=PatternType.CONTRADICTION,
            title="Medium",
            evidence=[
                EvidenceItem(
                    assertion_node_id="a0",
                    assertion_text="claim A",
                    source_document_id="d0",
                    source_url="https://a.example",
                    source_tier=1,
                )
            ],
            counter_evidence=[
                EvidenceItem(
                    assertion_node_id="a1",
                    assertion_text="claim B",
                    source_document_id="d1",
                    source_url="https://b.example",
                    source_tier=2,
                )
            ],
            confidence_score=0.6,
        )
        result = verify_pattern(pattern)
        assert result is not None
        assert result.withheld_reason is None
        assert result.confidence_level == ConfidenceLevel.HIGH

    def test_withholds_insufficient_evidence(self) -> None:
        pattern = PatternCandidate(
            pattern_type=PatternType.BRIDGE,
            title="Weak",
            evidence=_make_evidence(2),
            confidence_score=0.5,
        )
        result = verify_pattern(pattern)
        assert result is not None
        assert result.withheld_reason is not None

    def test_unresolved_with_counter_evidence(self) -> None:
        pattern = PatternCandidate(
            pattern_type=PatternType.BRIDGE,
            title="Unresolved",
            evidence=_make_evidence(5, [1, 2, 1, 2, 1]),
            counter_evidence=_make_evidence(1, [2]),
            confidence_score=0.5,
        )
        result = verify_pattern(pattern)
        assert result is not None
        assert result.confidence_level == ConfidenceLevel.UNRESOLVED

    def test_adds_default_blind_spot(self) -> None:
        pattern = PatternCandidate(
            pattern_type=PatternType.GAP,
            title="Test",
            evidence=_make_evidence(5, [1, 2, 1, 2, 1]),
            blind_spots=[],
            confidence_score=0.7,
        )
        result = verify_pattern(pattern)
        assert len(result.blind_spots) >= 1

    def test_never_crashes_on_edge_cases(self) -> None:
        empty = PatternCandidate(pattern_type=PatternType.BRIDGE, title="Empty")
        result = verify_pattern(empty)
        assert result is not None

    def test_verify_all_splits(self) -> None:
        strong = PatternCandidate(
            pattern_type=PatternType.BRIDGE,
            title="Strong",
            evidence=_make_evidence(5, [1, 2, 1, 2, 1]),
            confidence_score=0.8,
        )
        weak = PatternCandidate(
            pattern_type=PatternType.BRIDGE,
            title="Weak",
            evidence=_make_evidence(2),
            confidence_score=0.3,
        )
        promoted, exploratory = verify_all([strong, weak])
        assert len(promoted) >= 1
        assert len(exploratory) >= 1

    def test_interpret_hook_fills_promoted_text(self) -> None:
        pattern = PatternCandidate(
            pattern_type=PatternType.BRIDGE,
            title="X",
            evidence=_make_evidence(5, [1, 2, 1, 2, 1]),
            confidence_score=0.8,
        )
        out = verify_pattern(pattern, interpret=lambda p: f"note:{p.pattern_type.value}")
        assert out is not None
        assert out.interpretation == "note:bridge"
