"""Pattern detection engines."""

from src.core.patterns.bridges import detect_bridges
from src.core.patterns.contradictions import detect_contradictions, refine_contradiction_candidates
from src.core.patterns.drift import detect_drift
from src.core.patterns.gaps import detect_gaps

__all__ = [
    "detect_bridges",
    "detect_contradictions",
    "detect_drift",
    "detect_gaps",
    "refine_contradiction_candidates",
]
