"""Markdown report, evidence JSON, and D3 graph HTML."""

from __future__ import annotations

import json
import logging
from typing import Any

import networkx as nx

from src.core.graph import KnowledgeGraph
from src.core.types import CoverageReport, PatternType, PromotedPattern

logger = logging.getLogger(__name__)


def _primary_edge_type(edge_data: dict[str, Any]) -> str:
    et = edge_data.get("edge_type")
    if et:
        return str(et)
    edges = edge_data.get("edges") or []
    if edges and isinstance(edges[0], dict):
        return str(edges[0].get("edge_type", "co_occurs"))
    return "co_occurs"


def generate_pattern_report(
    promoted: list[PromotedPattern],
    exploratory: list[PromotedPattern],
    coverage: CoverageReport,
) -> str:
    sections = ["# Pattern Discovery Report\n"]
    sections.append(
        f"**Patterns found:** {len(promoted)} promoted, {len(exploratory)} exploratory\n"
    )

    if promoted:
        sections.append("## Promoted patterns\n")
        for p in promoted:
            sections.append(_format_pattern_card(p))

    if exploratory:
        sections.append("## Exploratory leads (not enough evidence to promote)\n")
        for p in exploratory:
            sections.append(_format_pattern_card(p, exploratory=True))

    sections.append("## Coverage\n")
    sections.append(f"- Sources used: {', '.join(coverage.source_families_used) or 'none'}\n")
    sections.append(
        f"- Sources missing: {', '.join(coverage.source_families_missing) or 'none identified'}\n"
    )
    sections.append(f"- Total documents: {coverage.total_documents}\n")
    if coverage.weak_subtopics:
        sections.append(f"- Weak subtopics: {', '.join(coverage.weak_subtopics)}\n")
    for note in coverage.notes:
        sections.append(f"- {note}\n")

    return "\n".join(sections)


def _format_pattern_card(p: PromotedPattern, exploratory: bool = False) -> str:
    lines = [f"### [{p.pattern_type.value.title()}]: {p.title}\n"]
    conf = p.confidence_level.value if p.confidence_level else "unscored"
    lines.append(f"**Confidence:** {conf}")
    if exploratory and p.withheld_reason:
        lines.append(f"**Withheld because:** {p.withheld_reason}")
    lines.append(f"\n**Measured pattern:** {p.measured_pattern}\n")

    if p.evidence:
        lines.append("**Supporting evidence:**")
        for e in p.evidence[:5]:
            lines.append(f"- (Tier {e.source_tier}) {e.assertion_text[:200]}")

    if p.counter_evidence:
        lines.append("\n**Counterevidence:**")
        for e in p.counter_evidence[:3]:
            lines.append(f"- (Tier {e.source_tier}) {e.assertion_text[:200]}")

    if p.interpretation:
        lines.append(f"\n**Interpretation:** {p.interpretation}")

    if p.blind_spots:
        lines.append("\n**Blind spots:**")
        for bs in p.blind_spots:
            lines.append(f"- [{bs.severity}] {bs.description}")

    lines.append("\n---\n")
    return "\n".join(lines)


def generate_evidence_table(
    promoted: list[PromotedPattern],
    exploratory: list[PromotedPattern],
) -> str:
    entries: list[dict[str, Any]] = []
    for status, patterns in (
        ("promoted", promoted),
        ("exploratory", exploratory),
    ):
        for p in patterns:
            base = {
                "pattern_id": p.id,
                "pattern_type": p.pattern_type.value,
                "pattern_title": p.title,
                "promotion_status": status,
                "confidence": p.confidence_level.value if p.confidence_level else "",
                "pattern_interpretation": p.interpretation or "",
                "withheld_reason": p.withheld_reason or "",
            }
            for e in p.evidence + p.counter_evidence:
                row = dict(base)
                row.update(
                    {
                        "assertion_text": e.assertion_text,
                        "source_url": e.source_url,
                        "source_tier": e.source_tier,
                        "role": e.role,
                    }
                )
                entries.append(row)
    return json.dumps(entries, indent=2)


def generate_patterns_summary(
    promoted: list[PromotedPattern],
    exploratory: list[PromotedPattern],
) -> str:
    """One JSON array of pattern records (promoted and exploratory) for APIs and dashboards."""

    def _one(p: PromotedPattern, status: str) -> dict[str, Any]:
        return {
            "id": p.id,
            "pattern_type": p.pattern_type.value,
            "title": p.title,
            "promotion_status": status,
            "confidence": p.confidence_level.value if p.confidence_level else "",
            "evidence_count": p.evidence_count,
            "counter_evidence_count": len(p.counter_evidence),
            "measured_pattern": p.measured_pattern or "",
            "interpretation": p.interpretation or "",
            "withheld_reason": p.withheld_reason or "",
            "promotion_reason": p.promotion_reason or "",
        }

    rows = [_one(p, "promoted") for p in promoted] + [_one(p, "exploratory") for p in exploratory]
    return json.dumps(rows, indent=2)


def generate_run_summary(snapshot: dict[str, Any]) -> str:
    """Serialize run metadata (inputs, resolved pack, pipeline stats). No API keys."""
    return json.dumps(snapshot, indent=2, default=str)


def generate_coverage_markdown(coverage: CoverageReport) -> str:
    lines = ["# Coverage report\n"]
    lines.append(f"- Total documents: {coverage.total_documents}")
    lines.append(f"- Source families used: {', '.join(coverage.source_families_used) or 'none'}")
    if coverage.documents_per_tier:
        lines.append(f"- Documents per tier: {coverage.documents_per_tier}")
    return "\n".join(lines) + "\n"


def generate_graph_html(
    graph: KnowledgeGraph,
    promoted: list[PromotedPattern],
    exploratory: list[PromotedPattern] | None = None,
) -> str:
    communities: dict[str, int] = {}
    try:
        comms = nx.community.louvain_communities(graph.g, seed=42)
        for i, comm in enumerate(comms):
            for node_id in comm:
                communities[node_id] = i
    except Exception as e:
        logger.debug("Louvain for viz skipped: %s", e)

    nodes_data: list[dict[str, Any]] = []
    for node in graph._node_registry.values():
        nodes_data.append(
            {
                "id": node.id,
                "name": node.name[:40],
                "type": node.node_type.value,
                "community": communities.get(node.id, 0),
                "description": (node.description or "")[:200],
            }
        )

    bridge_pairs: set[tuple[str, str]] = set()
    contradiction_pairs: set[tuple[str, str]] = set()
    for p in list(promoted) + list(exploratory or []):
        if p.pattern_type == PatternType.BRIDGE:
            nu = p.details.get("bridge_node_u", "")
            nv = p.details.get("bridge_node_v", "")
            if nu and nv:
                bridge_pairs.add((str(nu), str(nv)))
        elif p.pattern_type == PatternType.CONTRADICTION:
            for e in p.evidence:
                for ce in p.counter_evidence:
                    contradiction_pairs.add((e.assertion_node_id, ce.assertion_node_id))

    links_data: list[dict[str, Any]] = []
    for u, v, data in graph.g.edges(data=True):
        color = "#999"
        nu = graph.get_node(u)
        nv = graph.get_node(v)
        if nu and nv:
            pair = (nu.name, nv.name)
            rev = (nv.name, nu.name)
            if pair in bridge_pairs or rev in bridge_pairs:
                color = "#EF9F27"
            if (u, v) in contradiction_pairs or (v, u) in contradiction_pairs:
                color = "#E24B4A"
        links_data.append(
            {
                "source": u,
                "target": v,
                "type": _primary_edge_type(data),
                "color": color,
            }
        )

    nodes_data = nodes_data[:500]
    node_ids = {n["id"] for n in nodes_data}
    links_data = [l for l in links_data if l["source"] in node_ids and l["target"] in node_ids]
    links_data = links_data[:2000]

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
body {{ margin: 0; font-family: -apple-system, sans-serif; background: #fafafa; }}
svg {{ width: 100%; height: 100vh; }}
.node circle {{ stroke: #fff; stroke-width: 1.5px; cursor: pointer; }}
.link {{ stroke-opacity: 0.4; }}
#tooltip {{ position: absolute; background: white; border: 1px solid #ddd;
  padding: 8px 12px; border-radius: 6px; font-size: 13px; pointer-events: none;
  display: none; max-width: 300px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
</style></head><body>
<div id="tooltip"></div>
<svg></svg>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js"></script>
<script>
const nodes = {json.dumps(nodes_data)};
const links = {json.dumps(links_data)};
const colors = d3.scaleOrdinal(d3.schemeTableau10);
const svg = d3.select("svg");
const width = window.innerWidth, height = window.innerHeight;
const sim = d3.forceSimulation(nodes)
  .force("link", d3.forceLink(links).id(d=>d.id).distance(60))
  .force("charge", d3.forceManyBody().strength(-80))
  .force("center", d3.forceCenter(width/2, height/2));
const link = svg.selectAll(".link").data(links).join("line")
  .attr("class","link").attr("stroke",d=>d.color).attr("stroke-width",1.5);
const node = svg.selectAll(".node").data(nodes).join("g").attr("class","node")
  .call(d3.drag().on("start",ds).on("drag",dd).on("end",de));
node.append("circle").attr("r",6).attr("fill",d=>colors(d.community));
node.append("text").text(d=>d.name).attr("dx",10).attr("dy",4)
  .style("font-size","11px").style("fill","#555");
const tooltip = d3.select("#tooltip");
node.on("mouseover",(e,d)=>{{tooltip.style("display","block")
  .html("<b>"+d.name+"</b><br>"+d.type+"<br>"+d.description)
  .style("left",(e.pageX+10)+"px").style("top",(e.pageY-10)+"px")}})
  .on("mouseout",()=>tooltip.style("display","none"));
sim.on("tick",()=>{{
  link.attr("x1",d=>d.source.x).attr("y1",d=>d.source.y)
    .attr("x2",d=>d.target.x).attr("y2",d=>d.target.y);
  node.attr("transform",d=>"translate("+d.x+","+d.y+")");
}});
function ds(e,d){{if(!e.active)sim.alphaTarget(.3).restart();d.fx=d.x;d.fy=d.y}}
function dd(e,d){{d.fx=e.x;d.fy=e.y}}
function de(e,d){{if(!e.active)sim.alphaTarget(0);d.fx=null;d.fy=null}}
</script></body></html>"""
