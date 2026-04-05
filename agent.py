"""
Pattern Discovery Agent — RunForge entry point.

Phases 6–10 (graph, pattern mining, verifier, report) will wire the full pipeline.
For now this registers the agent and completes topic routing + logging.
"""

from __future__ import annotations

import logging

from agent_runtime import AgentRuntime

from src.packs.research import ResearchPack
from src.packs.research.router import build_source_plan

logger = logging.getLogger(__name__)

runtime = AgentRuntime()

STEPS = [
    "classify_topic",
    "route_sources",
    "ingest_corpus",
    "expand_corpus",
    "extract_knowledge",
    "build_graph",
    "mine_patterns",
    "verify_patterns",
    "generate_report",
]


@runtime.agent(name="pattern-discovery", planned_steps=STEPS)
async def run(ctx, input):  # noqa: ANN001
    pack = ResearchPack()
    payload = {**(input or {}), **ctx.inputs}
    topic = payload.get("topic", "")
    depth = payload.get("depth", "standard")
    focus = payload.get("focus", "all")
    time_range = payload.get("time_range")
    max_documents = int(payload.get("max_documents", 100))

    with ctx.safe_step("classify_topic"):
        ctx.log(f"Topic: {topic!r}, domain: {pack.domain}")
        ctx.state["topic"] = topic
        ctx.state["domain"] = pack.domain

    with ctx.safe_step("route_sources"):
        plan = build_source_plan(
            topic,
            depth=depth,
            focus=focus,
            time_range=time_range,
            max_documents=max_documents,
        )
        ctx.state["source_plan"] = {"connectors": plan.connectors, "queries": plan.queries}
        ctx.log(f"Source plan: connectors={plan.connectors}, queries={len(plan.queries)}")

    ctx.log(
        "Pipeline stages through extraction and graph build are implemented in src/; "
        "wire remaining safe_steps in a follow-up to run end-to-end.",
        level="warning",
    )

    sp = ctx.state.get("source_plan") or {}
    return {
        "status": "partial",
        "domain": pack.domain,
        "connectors": sp.get("connectors", []),
        "message": "Scaffold: classify + route complete; ingest through report not yet invoked from agent.py",
    }


if __name__ == "__main__":
    runtime.serve()
