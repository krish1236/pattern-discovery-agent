"""Fill research interpretation templates from pattern details and evidence."""

from __future__ import annotations

import json
from typing import Any

from src.core.types import PatternCandidate, PatternType


def _pick(*vals: Any) -> str:
    for v in vals:
        if v is not None and v != "":
            return str(v)
    return "?"


class _SafeFormat(dict[str, str]):
    """Leave unknown ``{placeholders}`` unchanged."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _stringify_context(raw: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in raw.items():
        if isinstance(v, (dict, list, tuple)):
            out[k] = json.dumps(v, default=str) if v is not None else ""
        elif v is None:
            out[k] = ""
        else:
            out[k] = str(v)
    return out


def format_research_interpretation(pattern: PatternCandidate, template: str) -> str:
    if not template:
        return ""

    ctx: dict[str, Any] = dict(pattern.details)
    ctx["title"] = pattern.title
    ctx["measured_pattern"] = pattern.measured_pattern or ""

    pt = pattern.pattern_type
    if pt == PatternType.BRIDGE:
        ctx.setdefault(
            "community_a",
            _pick(ctx.get("bridge_node_u"), ctx.get("community_u")),
        )
        ctx.setdefault(
            "community_b",
            _pick(ctx.get("bridge_node_v"), ctx.get("community_v")),
        )
        bu = _pick(ctx.get("bridge_node_u"))
        bv = _pick(ctx.get("bridge_node_v"))
        ctx.setdefault(
            "bridge_entity",
            f"{bu} ↔ {bv}" if bu != "?" and bv != "?" else pattern.title,
        )
    elif pt == PatternType.CONTRADICTION:
        ev, ce = pattern.evidence, pattern.counter_evidence
        if ev:
            ctx.setdefault("assertion_a", (ev[0].assertion_text or "")[:500])
            ctx.setdefault("source_a", ev[0].source_url or "unknown")
        if ce:
            ctx.setdefault("assertion_b", (ce[0].assertion_text or "")[:500])
            ctx.setdefault("source_b", ce[0].source_url or "unknown")
        elif len(ev) > 1:
            ctx.setdefault("assertion_b", (ev[1].assertion_text or "")[:500])
            ctx.setdefault("source_b", ev[1].source_url or "unknown")
    elif pt == PatternType.DRIFT:
        aw = ctx.get("all_windows")
        if isinstance(aw, (list, tuple)) and aw:
            ctx.setdefault("time_windows", ", ".join(str(x) for x in aw))
        else:
            ctx.setdefault(
                "time_windows",
                f'{ctx.get("from_window", "?")} → {ctx.get("to_window", "?")}',
            )
        tr = ctx.get("transitions")
        ctx.setdefault("transitions", json.dumps(tr, default=str) if tr is not None else "")
    elif pt == PatternType.GAP:
        if "recommender_count" not in ctx:
            rc = ctx.get("recommendation_count")
            ctx["recommender_count"] = "?" if rc is None else str(rc)
        if "execution_count" not in ctx:
            ec = ctx.get("execution_count")
            ctx["execution_count"] = "?" if ec is None else str(ec)
        ctx.setdefault("gap_description", ctx.get("gap_node") or pattern.title)

    try:
        return template.format_map(_SafeFormat(_stringify_context(ctx)))
    except ValueError:
        return template
