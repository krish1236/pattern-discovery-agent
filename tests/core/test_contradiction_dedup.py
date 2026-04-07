"""Contradiction post-verification dedup (embedding groups)."""

from unittest.mock import patch

import numpy as np

from src.core.patterns.dedup import dedup_promoted_contradictions
from src.core.types import EvidenceItem, PatternType, PromotedPattern


def _c(
    pid: str,
    sup: str,
    cnt: str,
    nli: float,
    url_a: str = "https://a.example",
    url_b: str = "https://b.example",
) -> PromotedPattern:
    return PromotedPattern(
        id=pid,
        pattern_type=PatternType.CONTRADICTION,
        title=f"T {pid}",
        evidence=[
            EvidenceItem("n1", sup, "d1", url_a, 2),
        ],
        counter_evidence=[
            EvidenceItem("n2", cnt, "d2", url_b, 2),
        ],
        confidence_score=nli * 0.8,
        details={"nli_confidence": nli},
    )


@patch("src.core.patterns.dedup.embed_texts")
def test_dedup_merges_near_duplicate_contradictions(mock_embed) -> None:
    # Identical rows for pattern 0 and 1 => all four cross-sims are 1.0.
    mock_embed.return_value = np.array(
        [
            [1.0, 0, 0, 0],
            [0, 1.0, 0, 0],
            [1.0, 0, 0, 0],
            [0, 1.0, 0, 0],
        ],
        dtype=np.float64,
    )

    p0 = _c("a", "Bitcoin does seven tps", "Bitcoin does thirteen tps", 0.7)
    p1 = _c("b", "Bitcoin does seven tps", "Bitcoin does thirteen tps", 0.9)
    bridge = PromotedPattern(
        id="br",
        pattern_type=PatternType.BRIDGE,
        title="Bridge",
        evidence=[EvidenceItem("x", "e", "", "https://z", 1)],
    )
    promoted = [bridge, p0, p1]

    out, stats = dedup_promoted_contradictions(promoted, similarity_threshold=0.8)
    assert stats["contradictions_before"] == 2
    assert stats["contradictions_after"] == 1
    assert stats["patterns_merged"] == 1
    assert len(out) == 2
    assert out[0].pattern_type == PatternType.BRIDGE
    keeper = out[1]
    assert keeper.id == "b"
    assert keeper.details.get("contradiction_dedup_group_size") == 2
    assert len(keeper.evidence) >= 1
    assert len(keeper.counter_evidence) >= 1


@patch("src.core.patterns.dedup.embed_texts")
def test_dedup_keeps_distant_pairs(mock_embed) -> None:
    dim = 8
    # Orthogonal supports for two patterns
    e0 = np.zeros((4, dim), dtype=np.float64)
    e0[0, 0] = 1.0
    e0[1, 1] = 1.0
    e0[2, 2] = 1.0
    e0[3, 3] = 1.0
    mock_embed.return_value = e0

    p0 = _c("a", "cats are mammals", "cats are fish", 0.9)
    p1 = _c("b", "python is slow", "python is fast", 0.85)
    out, stats = dedup_promoted_contradictions([p0, p1], similarity_threshold=0.8)
    assert stats["contradictions_after"] == 2
    assert len(out) == 2
