"""Corpus deduplication, expansion, and tier assignment."""

from __future__ import annotations

import logging
from difflib import SequenceMatcher

import numpy as np

from src.core.types import SourceDocument
from src.domain_pack import DomainPack, SourceConnector

logger = logging.getLogger(__name__)


def _normalize_title(title: str) -> str:
    return title.lower().strip().rstrip(".")


def _title_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _normalize_title(a), _normalize_title(b)).ratio()


def _get_dedup_key(doc: SourceDocument) -> str | None:
    doi = doc.metadata.get("doi", "")
    if doi:
        return f"doi:{str(doi).lower().strip()}"
    openalex_id = doc.metadata.get("openalex_id", "")
    if openalex_id:
        return f"oa:{openalex_id}"
    s2_id = doc.metadata.get("s2_id", "")
    if s2_id:
        return f"s2:{s2_id}"
    return None


def deduplicate(documents: list[SourceDocument], title_threshold: float = 0.9) -> list[SourceDocument]:
    seen_keys: dict[str, SourceDocument] = {}
    no_key: list[SourceDocument] = []

    for doc in documents:
        key = _get_dedup_key(doc)
        if key:
            if key in seen_keys:
                existing = seen_keys[key]
                if len(doc.abstract) > len(existing.abstract) or doc.source_tier < existing.source_tier:
                    seen_keys[key] = doc
            else:
                seen_keys[key] = doc
        else:
            no_key.append(doc)

    deduped_no_key: list[SourceDocument] = []
    for doc in no_key:
        is_dup = False
        for existing in list(seen_keys.values()) + deduped_no_key:
            if _title_similarity(doc.title, existing.title) > title_threshold:
                is_dup = True
                if len(doc.abstract) > len(existing.abstract):
                    if existing in deduped_no_key:
                        deduped_no_key.remove(existing)
                        deduped_no_key.append(doc)
                break
        if not is_dup:
            deduped_no_key.append(doc)

    result = list(seen_keys.values()) + deduped_no_key
    logger.info("Dedup: %s -> %s documents", len(documents), len(result))
    return result


def filter_by_relevance(
    documents: list[SourceDocument],
    topic: str,
    min_similarity: float = 0.25,
) -> list[SourceDocument]:
    """Drop documents whose title/abstract embedding is far from the topic embedding."""
    if not documents or not (topic or "").strip():
        return documents

    from src.shared.embeddings import cosine_similarity, embed_texts

    topic_emb = np.asarray(embed_texts([topic.strip()])[0], dtype=np.float64).ravel()
    texts = [(d.abstract or d.title or "")[:2000] for d in documents]
    doc_embs = embed_texts(texts)
    filtered: list[SourceDocument] = []
    for doc, emb in zip(documents, doc_embs, strict=True):
        e = np.asarray(emb, dtype=np.float64).ravel()
        if topic_emb.shape != e.shape or topic_emb.size == 0:
            filtered.append(doc)
            continue
        sim = cosine_similarity(topic_emb, e)
        if sim >= min_similarity:
            filtered.append(doc)
        else:
            logger.debug(
                "Dropping irrelevant doc (sim=%.2f): %s",
                sim,
                (doc.title or "")[:60],
            )
    dropped = len(documents) - len(filtered)
    if dropped:
        logger.info("Relevance filter: dropped %s/%s off-topic documents", dropped, len(documents))
    return filtered


def cap_documents_round_robin_by_family(documents: list[SourceDocument], max_n: int) -> list[SourceDocument]:
    """Truncate to max_n while round-robining across source_family (avoids scholarly-only caps)."""
    if len(documents) <= max_n:
        return documents

    from collections import defaultdict, deque

    buckets: dict[str, deque[SourceDocument]] = defaultdict(deque)
    family_order: list[str] = []
    seen_f: set[str] = set()
    for d in documents:
        fam = d.source_family or "unknown"
        if fam not in seen_f:
            family_order.append(fam)
            seen_f.add(fam)
        buckets[fam].append(d)

    out: list[SourceDocument] = []
    fi = 0
    while len(out) < max_n:
        progressed = False
        for _ in range(len(family_order)):
            fam = family_order[fi % len(family_order)]
            fi += 1
            if buckets[fam]:
                out.append(buckets[fam].popleft())
                progressed = True
                if len(out) >= max_n:
                    break
        if not progressed:
            break
    return out


async def expand_corpus(
    documents: list[SourceDocument],
    connectors: list[SourceConnector],
    budget: int = 50,
    existing_ids: set[str] | None = None,
) -> list[SourceDocument]:
    if existing_ids is None:
        existing_ids = {doc.source_id for doc in documents}

    sorted_docs = sorted(
        documents,
        key=lambda d: d.metadata.get("cited_by_count", 0) or d.metadata.get("citation_count", 0),
        reverse=True,
    )

    new_docs: list[SourceDocument] = []
    for doc in sorted_docs[:20]:
        if len(new_docs) >= budget:
            break
        for connector in connectors:
            try:
                expanded = await connector.expand(doc.source_id, limit=5)
                for new_doc in expanded:
                    if new_doc.source_id not in existing_ids and len(new_docs) < budget:
                        existing_ids.add(new_doc.source_id)
                        new_docs.append(new_doc)
            except Exception as e:
                logger.warning("Expansion failed for %s: %s", doc.source_id, e)
                continue

    logger.info("Expansion: added %s documents", len(new_docs))
    return new_docs


def assign_tiers(documents: list[SourceDocument], pack: DomainPack) -> list[SourceDocument]:
    for doc in documents:
        doc.source_tier = pack.classify_tier(doc)
    return documents


def corpus_stats(documents: list[SourceDocument]) -> dict:
    tier_counts: dict[int, int] = {}
    family_counts: dict[str, int] = {}
    for doc in documents:
        tier_counts[doc.source_tier] = tier_counts.get(doc.source_tier, 0) + 1
        family_counts[doc.source_family] = family_counts.get(doc.source_family, 0) + 1

    return {
        "total_documents": len(documents),
        "documents_per_tier": tier_counts,
        "documents_per_family": family_counts,
        "with_abstract": sum(1 for d in documents if d.abstract),
        "with_authors": sum(1 for d in documents if d.authors),
    }
