"""Gap detection: many incoming recommends, few evaluates/produces."""

from __future__ import annotations

import logging
from collections import defaultdict

from src.core.graph import KnowledgeGraph
from src.core.types import BlindSpot, EdgeType, EvidenceItem, NodeType, PatternCandidate, PatternType

logger = logging.getLogger(__name__)


def _recommender_source_key(graph: KnowledgeGraph, recommender_id: str) -> str:
    rec_node = graph.get_node(recommender_id)
    if not rec_node:
        return recommender_id
    return str(rec_node.properties.get("source_document_id") or recommender_id)


def detect_gaps(
    graph: KnowledgeGraph,
    min_recommendations: int = 2,
    top_k: int = 10,
) -> list[PatternCandidate]:
    recommends_count: dict[str, list[str]] = defaultdict(list)
    execution_count: dict[str, int] = defaultdict(int)

    for _u, _v, data in graph.g.edges(data=True):
        for edge_dict in data.get("edges", []):
            etype = edge_dict.get("edge_type", "")
            src = edge_dict.get("source_node_id")
            tgt = edge_dict.get("target_node_id")
            if not src or not tgt:
                continue
            if etype == EdgeType.RECOMMENDS.value:
                recommends_count[tgt].append(src)
            elif etype in (EdgeType.EVALUATES.value, EdgeType.PRODUCES.value):
                execution_count[tgt] += 1

    gaps: list[tuple[str, int, int]] = []
    for node_id, recommenders in recommends_count.items():
        node = graph.get_node(node_id)
        if not node or node.node_type not in (NodeType.CONCEPT, NodeType.ARTIFACT):
            continue
        name = (node.name or "").strip()
        if len(name) < 4 or len(name.split()) < 2:
            continue
        unique_sources = {_recommender_source_key(graph, rid) for rid in recommenders}
        n_sources = len(unique_sources)
        if n_sources >= min_recommendations:
            ex = execution_count.get(node_id, 0)
            if ex <= 1:
                gaps.append((node_id, n_sources, ex))

    gaps.sort(key=lambda x: x[1], reverse=True)
    gaps = gaps[:top_k]

    patterns: list[PatternCandidate] = []
    for node_id, rec_count, exec_count in gaps:
        node = graph.get_node(node_id)
        if not node:
            continue
        evidence: list[EvidenceItem] = []
        for recommender_id in recommends_count[node_id][:5]:
            rec_node = graph.get_node(recommender_id)
            if rec_node:
                evidence.append(
                    EvidenceItem(
                        assertion_node_id=rec_node.id,
                        assertion_text=f"Recommends: {node.name}",
                        source_document_id=str(rec_node.properties.get("source_document_id", "")),
                        source_url=str(rec_node.properties.get("source_url", "")),
                        source_tier=int(rec_node.properties.get("source_tier", 2)),
                    )
                )
        patterns.append(
            PatternCandidate(
                pattern_type=PatternType.GAP,
                title=f"Gap: {node.name}",
                measured_pattern=(
                    f"'{node.name}' is recommended by {rec_count} distinct source document(s) "
                    f"but has only {exec_count} execution or evaluation edges."
                ),
                evidence=evidence,
                blind_spots=[
                    BlindSpot(
                        description="Work may exist outside indexed sources.",
                        severity="moderate",
                    )
                ],
                confidence_score=min(rec_count / 5.0, 0.9),
                domain=node.domain,
                details={
                    "gap_node": node.name,
                    "recommendation_count": rec_count,
                    "execution_count": exec_count,
                },
            )
        )

    logger.info("Produced %s gap pattern candidates", len(patterns))
    return patterns
