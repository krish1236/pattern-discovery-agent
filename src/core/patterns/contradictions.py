"""Contradiction detection using embedding pre-filter + NLI cross-encoder."""

from __future__ import annotations

import json
import logging
import os
import re
from itertools import combinations
from typing import Any

import numpy as np

from src.core.graph import KnowledgeGraph
from src.core.types import BlindSpot, EvidenceItem, Node, NodeType, PatternCandidate, PatternType
from src.shared.embeddings import cosine_similarity

logger = logging.getLogger(__name__)

CONTRADICTION_REFINE_MODEL = os.environ.get(
    "ANTHROPIC_CONTRADICTION_MODEL", "claude-sonnet-4-6"
)

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
        row_f = row.astype(np.float64)
        exp_scores = np.exp(row_f - np.max(row_f))
        probs = exp_scores / exp_scores.sum()
        idx = int(np.argmax(probs))
        raw_label = labels[idx]
        norm = str(raw_label).lower()
        results.append(
            {
                "label": norm,
                "confidence": float(probs[idx]),
                "scores": {str(labels[i]).lower(): float(probs[i]) for i in range(len(labels))},
            }
        )
    return results


def _find_source_doc_for_node(graph: KnowledgeGraph, node: Node) -> tuple[str, str, int]:
    """Resolve source document id, URL, and tier for an assertion (properties then neighbors)."""
    src_id = str(node.properties.get("source_document_id", ""))
    src_url = str(node.properties.get("source_url", ""))
    src_tier = int(node.properties.get("source_tier", 2))
    if src_url:
        return src_id, src_url, src_tier
    for neighbor in graph.get_neighbors(node.id):
        if neighbor.node_type == NodeType.SOURCE_DOCUMENT:
            return (
                neighbor.id,
                str(neighbor.properties.get("source_url", "")),
                int(neighbor.properties.get("source_tier", 2)),
            )
    return "", "", 2


def _gather_corroborating_evidence(
    graph: KnowledgeGraph,
    node_a: Node,
    node_b: Node,
    max_per_side: int = 3,
) -> tuple[list[EvidenceItem], list[EvidenceItem]]:
    """Assertions embedding-near one side but not the other extend the contradiction chain."""
    all_assertions = graph.get_nodes_by_type(NodeType.ASSERTION)

    if not node_a.embedding or not node_b.embedding:
        return [], []

    emb_a = np.asarray(node_a.embedding, dtype=float)
    emb_b = np.asarray(node_b.embedding, dtype=float)

    exclude_ids = {node_a.id, node_b.id}
    candidates = [
        a
        for a in all_assertions
        if a.id not in exclude_ids
        and a.embedding
        and len(a.embedding) == len(node_a.embedding)
    ]

    supports_a: list[tuple[float, Node]] = []
    supports_b: list[tuple[float, Node]] = []

    for a in candidates:
        emb = np.asarray(a.embedding, dtype=float)
        sim_to_a = cosine_similarity(emb, emb_a)
        sim_to_b = cosine_similarity(emb, emb_b)

        if sim_to_a > sim_to_b + 0.05 and sim_to_a > 0.6:
            supports_a.append((sim_to_a, a))
        elif sim_to_b > sim_to_a + 0.05 and sim_to_b > 0.6:
            supports_b.append((sim_to_b, a))

    supports_a.sort(key=lambda x: x[0], reverse=True)
    supports_b.sort(key=lambda x: x[0], reverse=True)

    def _to_evidence(items: list[tuple[float, Node]], role: str) -> list[EvidenceItem]:
        result: list[EvidenceItem] = []
        seen_texts: set[str] = set()
        for _, n in items:
            text = (n.description or n.name or "")[:200]
            if not text or text in seen_texts:
                continue
            seen_texts.add(text)
            src = _find_source_doc_for_node(graph, n)
            result.append(
                EvidenceItem(
                    assertion_node_id=n.id,
                    assertion_text=text,
                    source_document_id=src[0],
                    source_url=src[1],
                    source_tier=src[2],
                    role=role,
                )
            )
            if len(result) >= max_per_side:
                break
        return result

    return _to_evidence(supports_a, "supports"), _to_evidence(supports_b, "counters")


def detect_contradictions(
    graph: KnowledgeGraph,
    similarity_threshold: float = 0.4,
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
        ea = np.asarray(a.embedding, dtype=float)
        eb = np.asarray(b.embedding, dtype=float)
        if ea.shape != eb.shape or ea.size == 0:
            continue
        sim = cosine_similarity(ea, eb)
        if sim >= similarity_threshold:
            candidate_pairs.append((a, b, float(sim)))

    candidate_pairs.sort(key=lambda x: x[2], reverse=True)
    candidate_pairs = candidate_pairs[:max_pairs]

    if not candidate_pairs:
        return []

    text_pairs = [(a.description or a.name, b.description or b.name) for a, b, _ in candidate_pairs]
    nli_results = _classify_nli(text_pairs)

    patterns: list[PatternCandidate] = []
    for (node_a, node_b, sim), nli_result in zip(candidate_pairs, nli_results):
        if nli_result["label"] != "contradiction":
            continue
        if nli_result["confidence"] < contradiction_confidence:
            continue

        src_a = _find_source_doc_for_node(graph, node_a)
        src_b = _find_source_doc_for_node(graph, node_b)

        pattern = PatternCandidate(
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
        extra_support, extra_counter = _gather_corroborating_evidence(
            graph, node_a, node_b, max_per_side=3
        )
        pattern.evidence.extend(extra_support)
        pattern.counter_evidence.extend(extra_counter)
        patterns.append(pattern)

    logger.info("Produced %s contradiction pattern candidates", len(patterns))
    return patterns


def _parse_refine_json(text: str) -> dict[str, Any]:
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```\s*$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            return json.loads(m.group(0))
        raise


async def refine_contradiction_candidates(
    candidates: list[PatternCandidate],
    contradiction_template: str,
    api_key: str,
) -> list[PatternCandidate]:
    """LLM second pass: filter apparent contradictions; annotate real/conditional."""
    if not api_key or not (contradiction_template or "").strip():
        return candidates

    import anthropic

    client = anthropic.AsyncAnthropic(api_key=api_key)
    refined: list[PatternCandidate] = []

    for p in candidates:
        if p.pattern_type != PatternType.CONTRADICTION:
            refined.append(p)
            continue

        tx_a = p.evidence[0].assertion_text if p.evidence else ""
        tx_b = p.counter_evidence[0].assertion_text if p.counter_evidence else ""
        url_a = p.evidence[0].source_url if p.evidence else "unknown"
        url_b = p.counter_evidence[0].source_url if p.counter_evidence else "unknown"

        scenario = contradiction_template.format(
            assertion_a=tx_a,
            assertion_b=tx_b,
            source_a=url_a,
            source_b=url_b,
        )
        user_msg = (
            f"{scenario}\n\n"
            "Classify this contradiction. Be precise:\n\n"
            "- 'real': The claims directly conflict and cannot both be true.\n"
            "  Examples: 'X achieves 95% accuracy' vs 'X accuracy is below 60%'.\n"
            "  'The system is secure' vs 'The system has known vulnerability Y'.\n"
            "  'Method scales to N' vs 'Method fails beyond M where M < N'.\n\n"
            "- 'conditional': Both claims could be true under different conditions.\n"
            "  Example: 'Works well for use case A' vs 'Fails for use case B'.\n"
            "  KEEP these — note the conditions.\n\n"
            "- 'apparent': Not actually contradictory — different topics, or one is a\n"
            "  subset/refinement of the other, or they use different definitions.\n"
            "  ONLY use this when there is genuinely no conflict.\n\n"
            "When in doubt between 'conditional' and 'apparent', choose 'conditional'.\n\n"
            "Respond ONLY with JSON (no markdown fences): "
            '{"classification":"real"|"conditional"|"apparent",'
            '"reasoning":"one or two sentences",'
            '"conditions_for_claim_a":"when claim A holds, if any",'
            '"conditions_for_claim_b":"when claim B holds, if any"}'
        )

        try:
            resp = await client.messages.create(
                model=CONTRADICTION_REFINE_MODEL,
                max_tokens=1024,
                messages=[{"role": "user", "content": user_msg}],
            )
            verdict = _parse_refine_json(resp.content[0].text)
        except (json.JSONDecodeError, anthropic.APIError, KeyError, IndexError, TypeError) as e:
            logger.warning("Contradiction LLM refine failed, keeping NLI candidate: %s", e)
            refined.append(p)
            continue

        cls_raw = str(verdict.get("classification", "")).lower().strip()
        if cls_raw in ("apparent", "non_contradiction", "not_contradictory", "none", "no"):
            logger.info("LLM classified contradiction as apparent; dropping candidate")
            continue

        reasoning = str(verdict.get("reasoning", "")).strip()
        cond_a = str(verdict.get("conditions_for_claim_a", "")).strip()
        cond_b = str(verdict.get("conditions_for_claim_b", "")).strip()

        details = dict(p.details or {})
        details["llm_contradiction_classification"] = cls_raw or "unknown"
        details["llm_contradiction_reasoning"] = reasoning
        if cond_a:
            details["conditions_for_claim_a"] = cond_a
        if cond_b:
            details["conditions_for_claim_b"] = cond_b

        suffix = f" LLM review ({cls_raw}): {reasoning}" if reasoning else f" LLM review: {cls_raw}."
        p.details = details
        p.measured_pattern = (p.measured_pattern or "") + suffix
        p.blind_spots = [
            *list(p.blind_spots or []),
            BlindSpot(
                description="LLM second pass: apparent vs real contradiction.",
                severity="minor",
            ),
        ]
        refined.append(p)

    return refined
