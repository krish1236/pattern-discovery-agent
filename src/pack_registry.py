"""Resolve a `DomainPack` from user input (extensible for future packs)."""

from __future__ import annotations

import logging
import re

from src.domain_pack import DomainPack
from src.packs.research import ResearchPack
from src.packs.research.router import is_research_topic

logger = logging.getLogger(__name__)

_ALIASES: dict[str, str] = {
    "": "research",
    "technical": "research",
    "papers": "research",
    "scholarly": "research",
    "science": "research",
    "default": "research",
}


def _normalize_domain_key(raw: str | None) -> str:
    if raw is None:
        return "research"
    s = raw.strip().lower()
    s = re.sub(r"[\s_]+", "-", s)
    return _ALIASES.get(s, s)


def resolve_domain_pack(
    *,
    domain: str | None = None,
    topic: str = "",
) -> DomainPack:
    """Return the pack for ``domain``, or infer from ``topic`` when unset.

    Unknown explicit domains fall back to research with a warning.
    """
    if domain and str(domain).strip():
        key = _normalize_domain_key(str(domain))
        if key == "research":
            return ResearchPack()
        logger.warning("No pack for domain %r; using research", key)
        return ResearchPack()

    if topic and is_research_topic(topic):
        return ResearchPack()

    if topic:
        logger.info(
            "Topic did not match research heuristics; using research pack (only pack implemented)."
        )
    return ResearchPack()
