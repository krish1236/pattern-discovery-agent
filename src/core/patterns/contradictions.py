"""Contradiction detection using embedding pre-filter + NLI cross-encoder."""

from __future__ import annotations

import logging
from itertools import combinations
from typing import Any

import numpy as np

from src.core.graph import KnowledgeGraph
from src.core.types import BlindSpot, EvidenceItem, Node, NodeType, PatternCandidate, PatternType
from src.shared.embeddings import cosine_similarity

logger = logging.getLogger(__name__)

_nli_model: Any = None


def _get_nli_model() -> Any:
    global _nli_model
    if _nli_model is None:
        from sentence_transformers import CrossEncoder

        _nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-base")
        logger.info("Loaded NLI model: cross-encoder/nli-deberta-v3-base")
    return _nli_model


def _nli_label_order(model: Any) -> list[str]:
    cfg = getattr(getattr(model, "model", None), "config", None)
    id2label = getattr(cfg, "id2label", None) if cfg is not None else None
    if isinstance(id2label, dict) and id2label:
        return [id2label[i] for i in sorted(id2label.keys())]
    return ["contradiction", "entailment", "neutral"]


def _classify_nli(pairs: list[tuple[str, str]]) -> list[dict[str, Any]]:
    model = _get_nli_model()
    raw = model.predict(pairs, show_progress_bar=False)
    arr = np.asarray(raw, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    labels = _nli_label_order(model)
    if arr.shape[1] != len(labels):
        labels = [f"label_{i}" for i in range(arr.shape[1])]

    results: list[dict[str, Any]] = []
    for row in arr:
        idx = int(np.argmax(row))
        raw_label = labels[idx]
        norm = str(raw_label).lower()
        results.append(
            {
                "label": norm,
                "confidence": float(row[idx]),
                "scores": {str(labels[i]).lower(): float(row[i]) for i in range(len(labels))},
            }
        )
    return results


def detect_contradictions(
    graph: KnowledgeGraph,
    similarity_threshold: float = 0.6,
    contradiction_confidence: float = 0.7,
    max_pairs: int = 2000,
) -> list[PatternCandidate]:
    assertions = graph.get_nodes_by_type(NodeType.ASSERTION)
    with_emb = [a for a in assertions if a.embedding is not None]
    if len(with_emb) < 2:
        logger.info("Not enough assertions with embeddings for contradiction detection")
        return []

    candidate_pairs: list[tuple[Node, Node, float]] = []
    for a, b in combinations(with_emb, 2):
        sim = cosine_similarity(np.asarray(a.embedding), np.asarray(b.embedding))
        if sim >= similarity_threshold:
            candidate_pairs.append((a, b, float(sim)))

    candidate_pairs.sort(key=lambda x: x[2], reverse=True)
    candidate_pairs = candidate_pairs[:max_pairs]

    if not candidate_pairs:
        return []

    text_pairs = [(a.description or a.name, b.description or b.name) for a, b, _ in candidate_pairs]
    nli_results = _classify_nli(text_pairs)

    def _find_source_doc(node: Node) -> tuple[str, str, int]:
        for neighbor in graph.get_neighbors(node.id):
            if neighbor.node_type == NodeType.SOURCE_DOCUMENT:
                return (
                    neighbor.id,
                    str(neighbor.properties.get("source_url", "")),
                    int(neighbor.properties.get("source_tier", 2)),
                )
        return ("", "", 2)

    patterns: list[PatternCandidate] = []
    for (node_a, node_b, sim), nli_result in zip(candidate_pairs, nli_results):
        if nli_result["label"] != "contradiction":
            continue
        if nli_result["confidence"] < contradiction_confidence:
            continue

        src_a = _find_source_doc(node_a)
        src_b = _find_source_doc(node_b)

        patterns.append(
            PatternCandidate(
                pattern_type=PatternType.CONTRADICTION,
                title=f"Contradiction: {node_a.name[:50]} vs {node_b.name[:50]}",
                measured_pattern=(
                    f"NLI classified assertions as contradictory "
                    f"(confidence={nli_result['confidence']:.2f}, cosine={sim:.2f})."
                ),
                evidence=[
                    EvidenceItem(
                        assertion_node_id=node_a.id,
                        assertion_text=(node_a.description or node_a.name)[:300],
                        source_document_id=src_a[0],
                        source_url=src_a[1],
                        source_tier=src_a[2],
                        role="supports",
                    )
                ],
                counter_evidence=[
                    EvidenceItem(
                        assertion_node_id=node_b.id,
                        assertion_text=(node_b.description or node_b.name)[:300],
                        source_document_id=src_b[0],
                        source_url=src_b[1],
                        source_tier=src_b[2],
                        role="counters",
                    )
                ],
                blind_spots=[
                    BlindSpot(
                        description="NLI may miss domain nuance; verify with domain experts.",
                        severity="moderate",
                    )
                ],
                confidence_score=nli_result["confidence"] * 0.8,
                domain=node_a.domain,
                details={
                    "assertion_a": node_a.description,
                    "assertion_b": node_b.description,
                    "nli_confidence": nli_result["confidence"],
                    "nli_scores": nli_result["scores"],
                    "cosine_similarity": sim,
                },
            )
        )

    logger.info("Produced %s contradiction pattern candidates", len(patterns))
    return patterns
