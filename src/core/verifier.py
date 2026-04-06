"""Promotion gate and confidence labels for pattern candidates."""

from __future__ import annotations

import logging
from collections.abc import Callable

from src.core.types import BlindSpot, ConfidenceLevel, PatternCandidate, PatternType, PromotedPattern

logger = logging.getLogger(__name__)

MIN_EVIDENCE_COUNT = 3
MIN_SOURCE_TIERS = 2
MIN_SOURCE_URLS = 3
MIN_CONFIDENCE = 0.4


def _check_evidence_sufficiency(pattern: PatternCandidate) -> str | None:
    """Returns a reason string if evidence is insufficient, None if OK."""

    if pattern.pattern_type == PatternType.CONTRADICTION:
        if not pattern.evidence or not pattern.counter_evidence:
            return "Contradiction needs both evidence and counter-evidence"
        ev_urls = {e.source_url for e in pattern.evidence if e.source_url}
        cev_urls = {e.source_url for e in pattern.counter_evidence if e.source_url}
        if not ev_urls or not cev_urls:
            return "Contradiction evidence missing source URLs"
        if ev_urls == cev_urls:
            return "Both sides of contradiction from same source"
        return None

    if pattern.pattern_type == PatternType.DRIFT:
        if pattern.evidence_count < 2:
            return f"Drift evidence count {pattern.evidence_count} < 2"
        return None

    if pattern.pattern_type == PatternType.GAP:
        if pattern.evidence_count < 2:
            return f"Gap evidence count {pattern.evidence_count} < 2"
        source_docs = {e.source_document_id for e in pattern.evidence if e.source_document_id}
        if len(source_docs) < 2:
            return "Gap recommendations from single source document"
        return None

    if pattern.evidence_count < MIN_EVIDENCE_COUNT:
        return f"Evidence count {pattern.evidence_count} < {MIN_EVIDENCE_COUNT}"
    return None


def _source_diversity_ok(pattern: PatternCandidate) -> bool:
    """Tier / URL diversity rules vary by pattern type."""

    if pattern.pattern_type == PatternType.CONTRADICTION:
        urls = {u for u in pattern.source_urls if u} | {
            e.source_url for e in pattern.counter_evidence if e.source_url
        }
        tiers = pattern.source_tiers | {e.source_tier for e in pattern.counter_evidence}
        return len(tiers) >= MIN_SOURCE_TIERS or len(urls) >= 2

    if pattern.pattern_type in (PatternType.DRIFT, PatternType.GAP):
        urls = {u for u in pattern.source_urls if u}
        tiers = pattern.source_tiers
        return (
            len(tiers) >= MIN_SOURCE_TIERS
            or len(urls) >= MIN_SOURCE_URLS
            or len(urls) >= 2
        )

    urls = {u for u in pattern.source_urls if u}
    tiers = pattern.source_tiers
    return len(tiers) >= MIN_SOURCE_TIERS or len(urls) >= MIN_SOURCE_URLS


def _compute_confidence_level(pattern: PatternCandidate) -> ConfidenceLevel:
    has_counter = len(pattern.counter_evidence) > 0
    ev_count = pattern.evidence_count

    if pattern.pattern_type == PatternType.CONTRADICTION:
        total_items = ev_count + len(pattern.counter_evidence)
        all_urls = pattern.source_urls | {
            e.source_url for e in pattern.counter_evidence if e.source_url
        }
        all_tiers = pattern.source_tiers | {e.source_tier for e in pattern.counter_evidence}
        has_tier_1 = 1 in all_tiers
        if total_items >= 2 and len(all_urls) >= 2 and len(all_tiers) >= 2:
            return ConfidenceLevel.HIGH if has_tier_1 else ConfidenceLevel.MEDIUM
        if total_items >= 2 and len(all_urls) >= 2:
            return ConfidenceLevel.MEDIUM
        return ConfidenceLevel.LOW

    has_tier_1 = 1 in pattern.source_tiers
    tier_count = len(pattern.source_tiers)

    if has_counter and pattern.confidence_score < 0.6:
        return ConfidenceLevel.UNRESOLVED

    if ev_count >= 5 and tier_count >= 2 and has_tier_1 and not has_counter:
        return ConfidenceLevel.HIGH

    if ev_count >= 3 and (tier_count >= 2 or len(pattern.source_urls) >= 3):
        return ConfidenceLevel.MEDIUM

    return ConfidenceLevel.LOW


def verify_pattern(
    pattern: PatternCandidate,
    *,
    interpret: Callable[[PatternCandidate], str] | None = None,
) -> PromotedPattern | None:
    reasons: list[str] = []

    evidence_issue = _check_evidence_sufficiency(pattern)
    if evidence_issue:
        reasons.append(evidence_issue)

    if not _source_diversity_ok(pattern):
        urls = {u for u in pattern.source_urls if u}
        tiers = pattern.source_tiers
        if pattern.pattern_type == PatternType.CONTRADICTION:
            urls |= {e.source_url for e in pattern.counter_evidence if e.source_url}
            tiers |= {e.source_tier for e in pattern.counter_evidence}
        reasons.append(
            f"Limited source diversity: {len(tiers)} tiers, {len(urls)} URLs"
        )

    if pattern.confidence_score < MIN_CONFIDENCE:
        reasons.append(f"Confidence {pattern.confidence_score:.2f} < {MIN_CONFIDENCE}")

    if not pattern.blind_spots:
        pattern.blind_spots.append(
            BlindSpot(
                description="No blind spots identified; default note added.",
                severity="minor",
            )
        )

    confidence_level = _compute_confidence_level(pattern)
    pattern.confidence_level = confidence_level

    if reasons:
        logger.info("Withholding pattern %r: %s", pattern.title, "; ".join(reasons))
        promoted = PromotedPattern(**pattern.__dict__)
        promoted.withheld_reason = "; ".join(reasons)
        promoted.confidence_level = ConfidenceLevel.LOW
        if interpret:
            promoted.interpretation = interpret(promoted)
        return promoted

    promoted = PromotedPattern(**pattern.__dict__)
    promoted.promotion_reason = "Passed all promotion checks"
    promoted.confidence_level = confidence_level
    if interpret:
        promoted.interpretation = interpret(promoted)
    logger.info("Promoted pattern %r with confidence=%s", pattern.title, confidence_level.value)
    return promoted


def verify_all(
    candidates: list[PatternCandidate],
    *,
    interpret: Callable[[PatternCandidate], str] | None = None,
) -> tuple[list[PromotedPattern], list[PromotedPattern]]:
    promoted: list[PromotedPattern] = []
    exploratory: list[PromotedPattern] = []

    for candidate in candidates:
        result = verify_pattern(candidate, interpret=interpret)
        if result:
            if result.withheld_reason:
                exploratory.append(result)
            else:
                promoted.append(result)

    logger.info("Verification: %s promoted, %s exploratory", len(promoted), len(exploratory))
    return promoted, exploratory
