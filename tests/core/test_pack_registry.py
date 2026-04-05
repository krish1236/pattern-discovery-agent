"""Domain pack resolution."""

from src.pack_registry import resolve_domain_pack
from src.packs.research import ResearchPack


def test_explicit_research_pack() -> None:
    p = resolve_domain_pack(domain="research", topic="")
    assert isinstance(p, ResearchPack)
    assert p.domain == "research"


def test_technical_alias() -> None:
    p = resolve_domain_pack(domain="technical", topic="")
    assert isinstance(p, ResearchPack)


def test_unknown_domain_falls_back_to_research(caplog) -> None:
    import logging

    caplog.set_level(logging.WARNING)
    p = resolve_domain_pack(domain="sports", topic="")
    assert isinstance(p, ResearchPack)
    assert any("No pack for domain" in r.message for r in caplog.records)


def test_topic_inference_research_keywords() -> None:
    p = resolve_domain_pack(domain=None, topic="neural architecture search trends")
    assert isinstance(p, ResearchPack)


def test_empty_uses_research() -> None:
    p = resolve_domain_pack(domain=None, topic="")
    assert isinstance(p, ResearchPack)
