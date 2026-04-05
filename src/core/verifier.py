"""Promotion gate and confidence labels for pattern candidates."""

from __future__ import annotations

import logging

from src.core.types import BlindSpot, ConfidenceLevel, PatternCandidate, PromotedPattern

logger = logging.getLogger(__name__)

MIN_EVIDENCE_COUNT = 3
MIN_SOURCE_TIERS = 2
MIN_SOURCE_URLS = 3
MIN_CONFIDENCE = 0.4


def _compute_confidence_level(pattern: PatternCandidate) -> ConfidenceLevel:
    has_tier_1 = 1 in pattern.source_tiers
    tier_count = len(pattern.source_tiers)
    ev_count = pattern.evidence_count
    has_counter = len(pattern.counter_evidence) > 0

    if has_counter and pattern.confidence_score < 0.6:
        return ConfidenceLevel.UNRESOLVED

    if ev_count >= 5 and tier_count >= 2 and has_tier_1 and not has_counter:
        return ConfidenceLevel.HIGH

    if ev_count >= 3 and (tier_count >= 2 or len(pattern.source_urls) >= 3):
        return ConfidenceLevel.MEDIUM

    return ConfidenceLevel.LOW


def verify_pattern(pattern: PatternCandidate) -> PromotedPattern | None:
    reasons: list[str] = []

    if pattern.evidence_count < MIN_EVIDENCE_COUNT:
        reasons.append(f"Evidence count {pattern.evidence_count} < {MIN_EVIDENCE_COUNT}")

    tier_diverse = len(pattern.source_tiers) >= MIN_SOURCE_TIERS
    url_diverse = len(pattern.source_urls) >= MIN_SOURCE_URLS
    if not (tier_diverse or url_diverse):
        reasons.append(
            f"Limited source diversity: {len(pattern.source_tiers)} tiers, "
            f"{len(pattern.source_urls)} URLs"
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
        return promoted

    promoted = PromotedPattern(**pattern.__dict__)
    promoted.promotion_reason = "Passed all promotion checks"
    promoted.confidence_level = confidence_level
    logger.info("Promoted pattern %r with confidence=%s", pattern.title, confidence_level.value)
    return promoted


def verify_all(
    candidates: list[PatternCandidate],
) -> tuple[list[PromotedPattern], list[PromotedPattern]]:
    promoted: list[PromotedPattern] = []
    exploratory: list[PromotedPattern] = []

    for candidate in candidates:
        result = verify_pattern(candidate)
        if result:
            if result.withheld_reason:
                exploratory.append(result)
            else:
                promoted.append(result)

    logger.info("Verification: %s promoted, %s exploratory", len(promoted), len(exploratory))
    return promoted, exploratory
