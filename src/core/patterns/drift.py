"""Temporal drift via yearly bins, embeddings per window, and HDBSCAN."""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np

from src.core.graph import KnowledgeGraph
from src.core.types import BlindSpot, EvidenceItem, Node, NodeType, PatternCandidate, PatternType
from src.shared.embeddings import embed_texts

logger = logging.getLogger(__name__)


def _bin_by_year(nodes: list) -> dict[str, list]:
    bins: dict[str, list] = defaultdict(list)
    for node in nodes:
        date_str = node.properties.get("publication_date", "")
        if date_str and len(str(date_str)) >= 4:
            year = str(date_str)[:4]
            bins[year].append(node)
    return dict(sorted(bins.items()))


def detect_drift(
    graph: KnowledgeGraph,
    min_cluster_size: int = 3,
    min_windows: int = 2,
) -> list[PatternCandidate]:
    docs = graph.get_nodes_by_type(NodeType.SOURCE_DOCUMENT)
    assertions = graph.get_nodes_by_type(NodeType.ASSERTION)
    dated_nodes = [n for n in docs + assertions if n.properties.get("publication_date")]
    bins = _bin_by_year(dated_nodes)

    if len(bins) < min_windows:
        logger.info("Only %s time windows, need >= %s", len(bins), min_windows)
        return []

    window_clusters: dict[str, list[set[int]]] = {}
    window_cluster_nodes: dict[str, dict[int, list[Node]]] = {}

    for window, nodes in bins.items():
        texts = [n.description or n.name for n in nodes]
        if len(texts) < min_cluster_size:
            continue
        try:
            embeddings = np.asarray(embed_texts(texts), dtype=np.float64)
            import hdbscan

            sim = np.dot(embeddings, embeddings.T)
            dist = (1.0 - np.clip(sim, -1.0, 1.0)).astype(np.float64, copy=False)
            np.fill_diagonal(dist, 0.0)
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="precomputed")
            labels = clusterer.fit_predict(dist)
            clusters: dict[int, set[int]] = defaultdict(set)
            for idx, label in enumerate(labels):
                if label >= 0:
                    clusters[label].add(idx)
            window_clusters[window] = list(clusters.values())
            window_cluster_nodes[window] = {
                label: [nodes[i] for i in sorted(indices)]
                for label, indices in clusters.items()
            }
        except Exception as e:
            logger.warning("HDBSCAN failed for window %s: %s", window, e)
            continue

    if len(window_clusters) < min_windows:
        return []

    windows_sorted = sorted(window_clusters.keys())
    transitions: list[dict] = []
    for i in range(1, len(windows_sorted)):
        prev_w = windows_sorted[i - 1]
        curr_w = windows_sorted[i]
        prev_count = len(window_clusters.get(prev_w, []))
        curr_count = len(window_clusters.get(curr_w, []))
        transitions.append(
            {
                "from_window": prev_w,
                "to_window": curr_w,
                "prev_clusters": prev_count,
                "curr_clusters": curr_count,
                "delta": curr_count - prev_count,
            }
        )

    patterns: list[PatternCandidate] = []
    for t in transitions:
        if abs(t["delta"]) < 1:
            continue
        direction = "grew" if t["delta"] > 0 else "consolidated"

        evidence: list[EvidenceItem] = []
        for window in (t["from_window"], t["to_window"]):
            cluster_content = window_cluster_nodes.get(window, {})
            for _label, cluster_nodes in cluster_content.items():
                assertion_nodes = [n for n in cluster_nodes if n.node_type == NodeType.ASSERTION]
                for node in assertion_nodes[:3]:
                    text = (node.description or node.name or "")[:200]
                    if not text:
                        continue
                    evidence.append(
                        EvidenceItem(
                            assertion_node_id=node.id,
                            assertion_text=text,
                            source_document_id=str(node.properties.get("source_document_id", "")),
                            source_url=str(node.properties.get("source_url", "")),
                            source_tier=int(node.properties.get("source_tier", 2)),
                        )
                    )

        patterns.append(
            PatternCandidate(
                pattern_type=PatternType.DRIFT,
                title=f"Topic drift: {t['from_window']} → {t['to_window']}",
                measured_pattern=(
                    f"Cluster count moved from {t['prev_clusters']} to {t['curr_clusters']} "
                    f"between {t['from_window']} and {t['to_window']} ({direction})."
                ),
                evidence=evidence[:10],
                blind_spots=[
                    BlindSpot(
                        description=f"Analysis covers {len(bins)} year bins only.",
                        severity="moderate",
                    )
                ],
                confidence_score=0.5,
                details={
                    "from_window": t["from_window"],
                    "to_window": t["to_window"],
                    "cluster_delta": t["delta"],
                    "all_windows": windows_sorted,
                    "transitions": transitions,
                },
            )
        )

    logger.info("Produced %s drift pattern candidates", len(patterns))
    return patterns
