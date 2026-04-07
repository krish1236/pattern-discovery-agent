"""Post-verification deduplication of promoted contradictions (embedding overlap)."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TypeVar

import numpy as np

from src.core.types import EvidenceItem, PatternCandidate, PatternType
from src.shared.embeddings import cosine_similarity, embed_texts

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=PatternCandidate)


class _UnionFind:
    def __init__(self, n: int) -> None:
        self._p = list(range(n))

    def find(self, x: int) -> int:
        while self._p[x] != x:
            self._p[x] = self._p[self._p[x]]
            x = self._p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._p[rb] = ra


def _nli_confidence(p: PatternCandidate) -> float:
    d = p.details or {}
    v = d.get("nli_confidence")
    if isinstance(v, (int, float)):
        return float(v)
    return float(p.confidence_score)


def _pair_assertion_max_sim(
    sup_i: np.ndarray,
    cnt_i: np.ndarray,
    sup_j: np.ndarray,
    cnt_j: np.ndarray,
) -> float:
    return max(
        cosine_similarity(sup_i, sup_j),
        cosine_similarity(cnt_i, cnt_j),
        cosine_similarity(sup_i, cnt_j),
        cosine_similarity(cnt_i, sup_j),
    )


def _dedupe_evidence_by_url(items: list[EvidenceItem]) -> list[EvidenceItem]:
    seen: set[str] = set()
    out: list[EvidenceItem] = []
    for it in items:
        key = (it.source_url or "").strip() or f"id:{it.assertion_node_id}"
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


def dedup_promoted_contradictions(
    promoted: list[T],
    *,
    similarity_threshold: float = 0.8,
) -> tuple[list[T], dict[str, int]]:
    """Merge near-duplicate promoted contradictions; drop merged rows from the list.

    Groups patterns when max cosine similarity across primary support/counter texts
    (all four cross-comparisons) exceeds ``similarity_threshold``.

    Returns:
        (new_promoted_list, stats) with keys:
        contradictions_before, contradictions_after, patterns_merged, groups_merged
    """
    if len(promoted) <= 1:
        n = sum(1 for p in promoted if p.pattern_type == PatternType.CONTRADICTION)
        return promoted, {
            "contradictions_before": n,
            "contradictions_after": n,
            "patterns_merged": 0,
            "groups_merged": 0,
        }

    contra_indices: list[int] = []
    for i, p in enumerate(promoted):
        if p.pattern_type != PatternType.CONTRADICTION:
            continue
        if not p.evidence or not p.counter_evidence:
            continue
        contra_indices.append(i)

    n_c = len(contra_indices)
    if n_c <= 1:
        n = sum(1 for p in promoted if p.pattern_type == PatternType.CONTRADICTION)
        return promoted, {
            "contradictions_before": n,
            "contradictions_after": n,
            "patterns_merged": 0,
            "groups_merged": 0,
        }

    patterns = [promoted[i] for i in contra_indices]
    texts: list[str] = []
    for p in patterns:
        texts.append((p.evidence[0].assertion_text or "")[:2000])
        texts.append((p.counter_evidence[0].assertion_text or "")[:2000])

    try:
        embs = np.asarray(embed_texts(texts), dtype=np.float64)
    except Exception as e:
        logger.warning("Contradiction dedup skipped (embed failed): %s", e)
        n = sum(1 for p in promoted if p.pattern_type == PatternType.CONTRADICTION)
        return promoted, {
            "contradictions_before": n,
            "contradictions_after": n,
            "patterns_merged": 0,
            "groups_merged": 0,
        }

    uf = _UnionFind(n_c)
    for i in range(n_c):
        sup_i, cnt_i = embs[2 * i], embs[2 * i + 1]
        for j in range(i + 1, n_c):
            sup_j, cnt_j = embs[2 * j], embs[2 * j + 1]
            if _pair_assertion_max_sim(sup_i, cnt_i, sup_j, cnt_j) > similarity_threshold:
                uf.union(i, j)

    comp_members: dict[int, list[int]] = defaultdict(list)
    for i in range(n_c):
        comp_members[uf.find(i)].append(i)

    keeper_by_root: dict[int, PatternCandidate] = {}
    merged_ids: set[str] = set()
    groups_merged = 0
    patterns_merged = 0

    for root, members in comp_members.items():
        best_local = max(members, key=lambda m: _nli_confidence(patterns[m]))
        keeper = patterns[best_local]
        all_ev: list[EvidenceItem] = []
        all_cev: list[EvidenceItem] = []
        merged_meta: list[dict[str, str]] = []

        for m in members:
            p = patterns[m]
            all_ev.extend(list(p.evidence))
            all_cev.extend(list(p.counter_evidence))
            if m != best_local:
                merged_meta.append({"id": p.id, "title": p.title})
                merged_ids.add(p.id)
                patterns_merged += 1

        keeper.evidence = _dedupe_evidence_by_url(all_ev)
        keeper.counter_evidence = _dedupe_evidence_by_url(all_cev)
        d = dict(keeper.details or {})
        if merged_meta:
            d["contradiction_dedup_merged"] = merged_meta
            d["contradiction_dedup_group_size"] = len(members)
            groups_merged += 1
        keeper.details = d
        keeper_by_root[root] = keeper

    id_to_local: dict[str, int] = {patterns[i].id: i for i in range(n_c)}

    out: list[T] = []
    emitted_roots: set[int] = set()

    for p in promoted:
        if p.pattern_type != PatternType.CONTRADICTION:
            out.append(p)
            continue
        if not p.evidence or not p.counter_evidence:
            out.append(p)
            continue
        lid = id_to_local.get(p.id)
        if lid is None:
            out.append(p)
            continue
        root = uf.find(lid)
        if p.id in merged_ids:
            continue
        if root in emitted_roots:
            continue
        emitted_roots.add(root)
        out.append(keeper_by_root[root])  # type: ignore[arg-type]

    c_after = sum(1 for p in out if p.pattern_type == PatternType.CONTRADICTION)
    c_before = sum(1 for p in promoted if p.pattern_type == PatternType.CONTRADICTION)

    return out, {
        "contradictions_before": c_before,
        "contradictions_after": c_after,
        "patterns_merged": patterns_merged,
        "groups_merged": groups_merged,
    }
