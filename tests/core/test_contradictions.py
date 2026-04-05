"""Contradiction / NLI tests (heavy tests are marked slow)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.patterns.contradictions import _classify_nli, refine_contradiction_candidates
from src.core.types import EvidenceItem, PatternCandidate, PatternType


@pytest.mark.slow
def test_nli_known_contradiction() -> None:
    pairs = [("The cat is sleeping", "The cat is not sleeping")]
    results = _classify_nli(pairs)
    assert results[0]["label"] == "contradiction"


@pytest.mark.slow
def test_nli_known_entailment() -> None:
    pairs = [("A man is eating pizza", "A man eats something")]
    results = _classify_nli(pairs)
    assert results[0]["label"] == "entailment"


@pytest.mark.slow
def test_nli_known_neutral() -> None:
    pairs = [("A man is eating pizza", "The weather is nice today")]
    results = _classify_nli(pairs)
    assert results[0]["label"] == "neutral"


@pytest.mark.asyncio
async def test_refine_contradiction_drops_apparent() -> None:
    tpl = (
        'Source A claims: "{assertion_a}" (from {source_a}). '
        'Source B claims: "{assertion_b}" (from {source_b}).'
    )
    p = PatternCandidate(
        pattern_type=PatternType.CONTRADICTION,
        title="C",
        measured_pattern="NLI flag",
        evidence=[
            EvidenceItem("a1", "claim A text", "d1", "https://a", 1),
        ],
        counter_evidence=[
            EvidenceItem("a2", "claim B text", "d2", "https://b", 1),
        ],
        confidence_score=0.8,
    )
    bridge = PatternCandidate(pattern_type=PatternType.BRIDGE, title="B", evidence=[], confidence_score=0.5)

    mock_msg = MagicMock()
    mock_msg.content = [MagicMock(text='{"classification":"apparent","reasoning":"Same method, different datasets."}')]
    mock_create = AsyncMock(return_value=mock_msg)

    with patch("anthropic.AsyncAnthropic") as mock_aclass:
        mock_aclass.return_value.messages.create = mock_create
        out = await refine_contradiction_candidates([bridge, p], tpl, "sk-test")
    assert len(out) == 1
    assert out[0].pattern_type == PatternType.BRIDGE
    mock_create.assert_awaited_once()


@pytest.mark.asyncio
async def test_refine_contradiction_keeps_real() -> None:
    tpl = (
        'A: "{assertion_a}" ({source_a}). B: "{assertion_b}" ({source_b}).'
    )
    p = PatternCandidate(
        pattern_type=PatternType.CONTRADICTION,
        title="C",
        measured_pattern="NLI",
        evidence=[EvidenceItem("a1", "x", "d1", "https://a", 1)],
        counter_evidence=[EvidenceItem("a2", "y", "d2", "https://b", 1)],
        confidence_score=0.8,
    )
    mock_msg = MagicMock()
    mock_msg.content = [MagicMock(text='{"classification":"real","reasoning":"Direct conflict on F1."}')]
    mock_create = AsyncMock(return_value=mock_msg)

    with patch("anthropic.AsyncAnthropic") as mock_aclass:
        mock_aclass.return_value.messages.create = mock_create
        out = await refine_contradiction_candidates([p], tpl, "sk-test")
    assert len(out) == 1
    assert out[0].details.get("llm_contradiction_classification") == "real"
    assert "LLM review" in out[0].measured_pattern
