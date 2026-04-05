"""LLM synthesis for pattern interpretations (report-ready prose)."""

from __future__ import annotations

import logging
import os

from src.core.types import PatternCandidate

logger = logging.getLogger(__name__)

INTERPRETATION_MODEL = os.environ.get("ANTHROPIC_INTERPRETATION_MODEL", "claude-sonnet-4-6")


def synthesize_pattern_interpretation(
    pattern: PatternCandidate,
    filled_template_context: str,
    *,
    api_key: str,
) -> str:
    """Turn filled template + evidence into short analyst-facing prose."""
    if not api_key or not (filled_template_context or "").strip():
        return filled_template_context

    import anthropic

    ev_snips: list[str] = []
    for e in pattern.evidence[:5]:
        t = (e.assertion_text or "").strip()
        if t:
            ev_snips.append(f"- (tier {e.source_tier}) {t[:280]}")
    evidence_block = "\n".join(ev_snips) if ev_snips else "(no evidence snippets)"

    user_msg = (
        "You are writing for a technical pattern-discovery report.\n\n"
        "**Structured context (from template fill):**\n"
        f"{filled_template_context}\n\n"
        "**Sample evidence:**\n"
        f"{evidence_block}\n\n"
        "Write 3–5 sentences of plain prose explaining why this pattern matters to a researcher "
        "surveying the literature. Answer any implicit questions (e.g. why a bridge matters); "
        "do not echo prompt instructions like 'Explain why…' as if addressing the reader. "
        "No markdown headings, no bullet lists."
    )

    try:
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=INTERPRETATION_MODEL,
            max_tokens=600,
            messages=[{"role": "user", "content": user_msg}],
        )
    except Exception as e:
        logger.warning("Interpretation LLM request failed: %s", e)
        return filled_template_context

    text = ""
    for block in getattr(resp, "content", []) or []:
        if isinstance(block, dict) and block.get("type") == "text":
            text += str(block.get("text", ""))
            continue
        t = getattr(block, "text", None)
        if t:
            text += t
    out = text.strip()
    return out if out else filled_template_context
