"""Bridge pattern detection (Louvain communities + semantic similarity on cross edges)."""

from __future__ import annotations

import logging

import networkx as nx
import numpy as np

from src.core.graph import KnowledgeGraph
from src.core.types import BlindSpot, EvidenceItem, NodeType, PatternCandidate, PatternType
from src.shared.embeddings import cosine_similarity

logger = logging.getLogger(__name__)


def detect_bridges(
    graph: KnowledgeGraph,
    min_community_size: int = 3,
    semantic_threshold: float = 0.4,
    top_k: int = 10,
) -> list[PatternCandidate]:
    if graph.node_count < 6:
        logger.info("Graph too small for bridge detection")
        return []

    communities = nx.community.louvain_communities(graph.g, resolution=1.0, seed=42)
    communities = [c for c in communities if len(c) >= min_community_size]

    if len(communities) < 2:
        logger.info("Only %s qualifying communities, need >= 2", len(communities))
        return []

    node_to_community: dict[str, int] = {}
    for i, comm in enumerate(communities):
        for node_id in comm:
            node_to_community[node_id] = i

    inter_community_edges: list[tuple[str, str, int, int]] = []
    for u, v in graph.g.edges():
        cu = node_to_community.get(u)
        cv = node_to_community.get(v)
        if cu is not None and cv is not None and cu != cv:
            inter_community_edges.append((u, v, cu, cv))

    if not inter_community_edges:
        return []

    bridge_candidates: list[dict] = []
    for u, v, cu, cv in inter_community_edges:
        node_u = graph.get_node(u)
        node_v = graph.get_node(v)
        if not node_u or not node_v:
            continue
        if not node_u.embedding or not node_v.embedding:
            continue
        sim = cosine_similarity(np.asarray(node_u.embedding), np.asarray(node_v.embedding))
        if sim >= semantic_threshold:
            bridge_candidates.append(
                {
                    "node_u": node_u,
                    "node_v": node_v,
                    "community_u": cu,
                    "community_v": cv,
                    "similarity": float(sim),
                }
            )

    if not bridge_candidates:
        return []

    centrality = nx.betweenness_centrality(graph.g)
    for bc in bridge_candidates:
        bc["centrality"] = max(
            centrality.get(bc["node_u"].id, 0.0),
            centrality.get(bc["node_v"].id, 0.0),
        )

    bridge_candidates.sort(
        key=lambda x: (x["similarity"] * 0.6 + x["centrality"] * 0.4),
        reverse=True,
    )

    patterns: list[PatternCandidate] = []
    for bc in bridge_candidates[:top_k]:
        nu = bc["node_u"]
        nv = bc["node_v"]
        evidence: list[EvidenceItem] = []
        for neighbor in graph.get_neighbors(nu.id):
            if neighbor.node_type == NodeType.ASSERTION:
                evidence.append(
                    EvidenceItem(
                        assertion_node_id=neighbor.id,
                        assertion_text=neighbor.description[:200],
                        source_document_id=str(neighbor.properties.get("source_document_id", "")),
                        source_url=str(neighbor.properties.get("source_url", "")),
                        source_tier=int(neighbor.properties.get("source_tier", 2)),
                    )
                )
        for neighbor in graph.get_neighbors(nv.id):
            if neighbor.node_type == NodeType.ASSERTION:
                evidence.append(
                    EvidenceItem(
                        assertion_node_id=neighbor.id,
                        assertion_text=neighbor.description[:200],
                        source_document_id=str(neighbor.properties.get("source_document_id", "")),
                        source_url=str(neighbor.properties.get("source_url", "")),
                        source_tier=int(neighbor.properties.get("source_tier", 2)),
                    )
                )

        patterns.append(
            PatternCandidate(
                pattern_type=PatternType.BRIDGE,
                title=f"Bridge: {nu.name} ↔ {nv.name}",
                measured_pattern=(
                    f"Nodes '{nu.name}' (community {bc['community_u']}) and "
                    f"'{nv.name}' (community {bc['community_v']}) link communities with "
                    f"cosine similarity {bc['similarity']:.2f} and betweenness "
                    f"{bc['centrality']:.3f}."
                ),
                evidence=evidence[:10],
                blind_spots=[
                    BlindSpot(
                        description=f"{len(evidence)} evidence items near bridge endpoints",
                        severity="moderate" if len(evidence) >= 3 else "major",
                    )
                ],
                confidence_score=min(bc["similarity"], 0.95),
                domain=nu.domain,
                details={
                    "community_u": bc["community_u"],
                    "community_v": bc["community_v"],
                    "bridge_node_u": nu.name,
                    "bridge_node_v": nv.name,
                    "semantic_similarity": bc["similarity"],
                    "betweenness_centrality": bc["centrality"],
                },
            )
        )

    logger.info("Produced %s bridge pattern candidates", len(patterns))
    return patterns
