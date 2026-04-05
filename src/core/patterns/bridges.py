"""Bridge pattern detection (Louvain communities + semantic similarity on cross edges)."""

from __future__ import annotations

import logging

import networkx as nx
import numpy as np

from src.core.graph import KnowledgeGraph
from src.core.types import BlindSpot, EvidenceItem, Node, NodeType, PatternCandidate, PatternType
from src.shared.embeddings import cosine_similarity

logger = logging.getLogger(__name__)

# Evidence: mention-first per endpoint, then embedding; cap merged list
_BRIDGE_EVIDENCE_PER_ENDPOINT = 5
_BRIDGE_EVIDENCE_MAX_TOTAL = 10
_MAX_ENDPOINT_BRIDGE_APPEARANCES = 2
_BRIDGE_PAIR_DEDUP_SIM = 0.7


def _collect_bridge_evidence(
    graph: KnowledgeGraph,
    bridge_node: Node,
    community_node_ids: set[str],
    max_items: int = 5,
) -> list[EvidenceItem]:
    """Assertions mentioning the bridge concept first, then by embedding similarity."""
    all_assertions: list[Node] = []
    for nid in community_node_ids:
        node = graph.get_node(nid)
        if node and node.node_type == NodeType.ASSERTION:
            all_assertions.append(node)

    bridge_name_lower = (bridge_node.name or "").lower()
    words = [w for w in bridge_name_lower.split() if len(w) > 3]

    by_mention: list[Node] = []
    by_embedding: list[Node] = []
    for a in all_assertions:
        text = ((a.description or "") + " " + (a.name or "")).lower()
        if bridge_name_lower and bridge_name_lower in text:
            by_mention.append(a)
        elif words and any(w in text for w in words):
            by_mention.append(a)
        else:
            by_embedding.append(a)

    if bridge_node.embedding and by_embedding:
        bridge_emb = np.asarray(bridge_node.embedding, dtype=float)
        scored: list[tuple[float, Node]] = []
        for a in by_embedding:
            if a.embedding and len(a.embedding) == len(bridge_node.embedding):
                sim = cosine_similarity(bridge_emb, np.asarray(a.embedding, dtype=float))
                scored.append((a, sim))
            else:
                scored.append((a, -1.0))
        scored.sort(key=lambda x: x[1], reverse=True)
        by_embedding = [a for a, _ in scored]

    ordered = by_mention + by_embedding
    evidence: list[EvidenceItem] = []
    seen_texts: set[str] = set()
    for a in ordered[: max_items * 2]:
        text = (a.description or a.name or "")[:200]
        if not text or text in seen_texts:
            continue
        seen_texts.add(text)
        evidence.append(
            EvidenceItem(
                assertion_node_id=a.id,
                assertion_text=text,
                source_document_id=str(a.properties.get("source_document_id", "")),
                source_url=str(a.properties.get("source_url", "")),
                source_tier=int(a.properties.get("source_tier", 2)),
            )
        )
        if len(evidence) >= max_items:
            break
    return evidence


def _merge_bridge_evidence(
    graph: KnowledgeGraph,
    items_u: list[EvidenceItem],
    items_v: list[EvidenceItem],
    nu: Node,
    nv: Node,
    comm_u: set[str],
    comm_v: set[str],
) -> tuple[list[EvidenceItem], int]:
    """Merge per-side evidence, dedupe by assertion id, cap total; fallback if empty."""
    seen_ids: set[str] = set()
    merged: list[EvidenceItem] = []
    for item in items_u + items_v:
        if item.assertion_node_id in seen_ids:
            continue
        seen_ids.add(item.assertion_node_id)
        merged.append(item)
        if len(merged) >= _BRIDGE_EVIDENCE_MAX_TOTAL:
            break

    raw_count = 0
    for nid in comm_u | comm_v:
        n = graph.get_node(nid)
        if n and n.node_type == NodeType.ASSERTION:
            raw_count += 1

    if merged:
        return merged[:_BRIDGE_EVIDENCE_MAX_TOTAL], raw_count

    # Fallback: any community assertions, ranked by mean bridge embedding
    fallback: list[EvidenceItem] = []
    for nid in comm_u | comm_v:
        node = graph.get_node(nid)
        if not node or node.node_type != NodeType.ASSERTION:
            continue
        text = (node.description or node.name or "")[:200]
        fallback.append(
            EvidenceItem(
                assertion_node_id=node.id,
                assertion_text=text,
                source_document_id=str(node.properties.get("source_document_id", "")),
                source_url=str(node.properties.get("source_url", "")),
                source_tier=int(node.properties.get("source_tier", 2)),
            )
        )
    ranked = _rank_evidence_by_bridge_embedding(graph, fallback, nu, nv)
    return ranked[:_BRIDGE_EVIDENCE_MAX_TOTAL], raw_count


def _rank_evidence_by_bridge_embedding(
    graph: KnowledgeGraph,
    evidence: list[EvidenceItem],
    nu: Node,
    nv: Node,
) -> list[EvidenceItem]:
    if not evidence:
        return evidence
    if not nu.embedding or not nv.embedding:
        return evidence[:_BRIDGE_EVIDENCE_MAX_TOTAL]
    eu = np.asarray(nu.embedding, dtype=float)
    ev = np.asarray(nv.embedding, dtype=float)
    if eu.shape != ev.shape or eu.size == 0:
        return evidence[:_BRIDGE_EVIDENCE_MAX_TOTAL]
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
    return [x[1] for x in scored[:_BRIDGE_EVIDENCE_MAX_TOTAL]]


def _dedupe_near_duplicate_bridges(candidates: list[dict]) -> list[dict]:
    """Drop candidates whose endpoint pairs are nearly the same (embedding space)."""
    out: list[dict] = []
    for bc in candidates:
        is_dup = False
        for existing in out:
            eu, ev = existing["node_u"], existing["node_v"]
            nu, nv = bc["node_u"], bc["node_v"]
            if not (
                nu.embedding
                and nv.embedding
                and eu.embedding
                and ev.embedding
            ):
                continue
            sim_uu = cosine_similarity(
                np.asarray(nu.embedding, dtype=float),
                np.asarray(eu.embedding, dtype=float),
            )
            sim_vv = cosine_similarity(
                np.asarray(nv.embedding, dtype=float),
                np.asarray(ev.embedding, dtype=float),
            )
            sim_uv = cosine_similarity(
                np.asarray(nu.embedding, dtype=float),
                np.asarray(ev.embedding, dtype=float),
            )
            sim_vu = cosine_similarity(
                np.asarray(nv.embedding, dtype=float),
                np.asarray(eu.embedding, dtype=float),
            )
            if (sim_uu > _BRIDGE_PAIR_DEDUP_SIM and sim_vv > _BRIDGE_PAIR_DEDUP_SIM) or (
                sim_uv > _BRIDGE_PAIR_DEDUP_SIM and sim_vu > _BRIDGE_PAIR_DEDUP_SIM
            ):
                is_dup = True
                break
        if not is_dup:
            out.append(bc)
    return out


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

    bridge_candidates = _dedupe_near_duplicate_bridges(hub_filtered)[:top_k]

    patterns: list[PatternCandidate] = []
    for bc in bridge_candidates:
        nu = bc["node_u"]
        nv = bc["node_v"]
        comm_u = communities[bc["community_u"]]
        comm_v = communities[bc["community_v"]]
        set_u = set(comm_u)
        set_v = set(comm_v)
        ev_u = _collect_bridge_evidence(
            graph, nu, set_u, max_items=_BRIDGE_EVIDENCE_PER_ENDPOINT
        )
        ev_v = _collect_bridge_evidence(
            graph, nv, set_v, max_items=_BRIDGE_EVIDENCE_PER_ENDPOINT
        )
        evidence, evidence_raw_count = _merge_bridge_evidence(
            graph, ev_u, ev_v, nu, nv, set_u, set_v
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
                evidence=evidence,
                blind_spots=[
                    BlindSpot(
                        description=(
                            f"{evidence_raw_count} community assertions; showing up to "
                            f"{len(evidence)} (mention-first per endpoint, else embedding-ranked)"
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
