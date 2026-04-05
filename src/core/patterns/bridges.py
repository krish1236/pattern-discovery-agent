"""Bridge pattern detection (Louvain communities + semantic similarity on cross edges)."""

from __future__ import annotations

import logging

import networkx as nx
import numpy as np

from src.core.graph import KnowledgeGraph
from src.core.types import BlindSpot, EvidenceItem, NodeType, PatternCandidate, PatternType
from src.shared.embeddings import cosine_similarity

logger = logging.getLogger(__name__)

# Evidence shown per bridge: ranked by embedding similarity to bridge pair (not arbitrary community order)
_BRIDGE_EVIDENCE_TOP_K = 5
_MAX_ENDPOINT_BRIDGE_APPEARANCES = 2


def _evidence_assertions_in_community(
    graph: KnowledgeGraph,
    community_node_ids: set[str],
) -> list[EvidenceItem]:
    """Collect assertion nodes from a Louvain community (not just graph neighbors of the bridge)."""
    out: list[EvidenceItem] = []
    for node_id in community_node_ids:
        node = graph.get_node(node_id)
        if not node or node.node_type != NodeType.ASSERTION:
            continue
        text = (node.description or node.name or "")[:200]
        out.append(
            EvidenceItem(
                assertion_node_id=node.id,
                assertion_text=text,
                source_document_id=str(node.properties.get("source_document_id", "")),
                source_url=str(node.properties.get("source_url", "")),
                source_tier=int(node.properties.get("source_tier", 2)),
            )
        )
    return out


def _rank_evidence_by_bridge_embedding(
    graph: KnowledgeGraph,
    evidence: list[EvidenceItem],
    nu,
    nv,
) -> list[EvidenceItem]:
    """Prefer assertions whose embeddings align with the bridge endpoint pair."""
    if not evidence:
        return evidence
    if not nu.embedding or not nv.embedding:
        return evidence[:_BRIDGE_EVIDENCE_TOP_K]
    eu = np.asarray(nu.embedding, dtype=float)
    ev = np.asarray(nv.embedding, dtype=float)
    if eu.shape != ev.shape or eu.size == 0:
        return evidence[:_BRIDGE_EVIDENCE_TOP_K]
    bridge_emb = (eu + ev) / 2.0

    scored: list[tuple[float, EvidenceItem]] = []
    for item in evidence:
        an = graph.get_node(item.assertion_node_id)
        if not an or not an.embedding:
            scored.append((-1.0, item))
            continue
        aemb = np.asarray(an.embedding, dtype=float)
        if aemb.shape != bridge_emb.shape:
            scored.append((-1.0, item))
            continue
        scored.append((cosine_similarity(bridge_emb, aemb), item))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [x[1] for x in scored[:_BRIDGE_EVIDENCE_TOP_K]]


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
        if (
            node_u.node_type == NodeType.SOURCE_DOCUMENT
            or node_v.node_type == NodeType.SOURCE_DOCUMENT
        ):
            continue
        if node_u.node_type != NodeType.CONCEPT or node_v.node_type != NodeType.CONCEPT:
            continue
        if not node_u.embedding or not node_v.embedding:
            continue
        eu = np.asarray(node_u.embedding, dtype=float)
        ev = np.asarray(node_v.embedding, dtype=float)
        if eu.shape != ev.shape or eu.size == 0:
            continue
        sim = cosine_similarity(eu, ev)
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

    endpoint_count: dict[str, int] = {}
    hub_filtered: list[dict] = []
    for bc in bridge_candidates:
        u_name, v_name = bc["node_u"].name, bc["node_v"].name
        if (
            endpoint_count.get(u_name, 0) >= _MAX_ENDPOINT_BRIDGE_APPEARANCES
            or endpoint_count.get(v_name, 0) >= _MAX_ENDPOINT_BRIDGE_APPEARANCES
        ):
            continue
        endpoint_count[u_name] = endpoint_count.get(u_name, 0) + 1
        endpoint_count[v_name] = endpoint_count.get(v_name, 0) + 1
        hub_filtered.append(bc)

    bridge_candidates = hub_filtered[:top_k]

    patterns: list[PatternCandidate] = []
    for bc in bridge_candidates:
        nu = bc["node_u"]
        nv = bc["node_v"]
        comm_u = communities[bc["community_u"]]
        comm_v = communities[bc["community_v"]]
        evidence = _evidence_assertions_in_community(graph, comm_u) + _evidence_assertions_in_community(
            graph, comm_v
        )
        seen_ids: set[str] = set()
        deduped: list[EvidenceItem] = []
        for item in evidence:
            if item.assertion_node_id in seen_ids:
                continue
            seen_ids.add(item.assertion_node_id)
            deduped.append(item)
        evidence_raw_count = len(deduped)
        evidence = _rank_evidence_by_bridge_embedding(graph, deduped, nu, nv)

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
                evidence=evidence,
                blind_spots=[
                    BlindSpot(
                        description=(
                            f"{evidence_raw_count} community assertions; showing top "
                            f"{len(evidence)} by embedding similarity to the bridge pair"
                        ),
                        severity="moderate" if evidence_raw_count >= 3 else "major",
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
