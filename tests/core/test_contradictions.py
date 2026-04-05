"""Contradiction / NLI tests (heavy tests are marked slow)."""

import pytest

from src.core.patterns.contradictions import _classify_nli


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
