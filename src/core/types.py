"""
Universal type system for the Pattern Discovery Engine.

Domain packs map external data into these types. The pattern engine, graph
builder, verifier, and report generator only use these abstractions.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum, IntEnum
from typing import Any


class NodeType(str, Enum):
    SOURCE_DOCUMENT = "source_document"
    ASSERTION = "assertion"
    ACTOR = "actor"
    CONCEPT = "concept"
    ARTIFACT = "artifact"
    METRIC = "metric"
    EVENT = "event"


class EdgeType(str, Enum):
    ASSERTS = "asserts"
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    INVOLVES = "involves"
    PRODUCES = "produces"
    EVALUATES = "evaluates"
    ASSOCIATED_WITH = "associated_with"
    EXTENDS = "extends"
    CO_OCCURS = "co_occurs"
    PRECEDES = "precedes"
    BRIDGES_TO = "bridges_to"
    RECOMMENDS = "recommends"


class SourceTier(IntEnum):
    PRIMARY = 1
    STRONG_SECONDARY = 2
    WEAK_SECONDARY = 3
    SOCIAL = 4


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNRESOLVED = "unresolved"


class PatternType(str, Enum):
    BRIDGE = "bridge"
    CONTRADICTION = "contradiction"
    DRIFT = "drift"
    GAP = "gap"


@dataclass
class EdgeMeta:
    source_tier: int
    source_family: str
    source_url: str
    source_id: str
    extraction_confidence: float
    extraction_model: str
    timestamp: str
    provenance: str
    domain: str

    def __post_init__(self) -> None:
        if not 1 <= self.source_tier <= 4:
            raise ValueError(f"source_tier must be 1-4, got {self.source_tier}")
        if not 0.0 <= self.extraction_confidence <= 1.0:
            raise ValueError(
                f"extraction_confidence must be 0.0-1.0, got {self.extraction_confidence}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EdgeMeta:
        return cls(**d)


@dataclass
class Node:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    node_type: NodeType = NodeType.CONCEPT
    name: str = ""
    description: str = ""
    domain: str = ""
    properties: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["node_type"] = self.node_type.value
        d.pop("embedding", None)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Node:
        data = dict(d)
        data["node_type"] = NodeType(data["node_type"])
        data.pop("embedding", None)
        return cls(**data)


@dataclass
class Edge:
    source_node_id: str
    target_node_id: str
    edge_type: EdgeType
    meta: EdgeMeta
    properties: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["edge_type"] = self.edge_type.value
        d["meta"] = self.meta.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Edge:
        data = dict(d)
        data["edge_type"] = EdgeType(data["edge_type"])
        data["meta"] = EdgeMeta.from_dict(data["meta"])
        return cls(**data)


@dataclass
class SourceDocument:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str = ""
    source_family: str = ""
    source_url: str = ""
    title: str = ""
    abstract: str = ""
    full_text: str | None = None
    authors: list[str] = field(default_factory=list)
    publication_date: str | None = None
    source_tier: int = 2
    domain: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    precomputed_embedding: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("precomputed_embedding", None)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SourceDocument:
        data = dict(d)
        data.pop("precomputed_embedding", None)
        return cls(**data)


@dataclass
class ExtractionResult:
    source_document_id: str
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)


@dataclass
class EvidenceItem:
    assertion_node_id: str
    assertion_text: str
    source_document_id: str
    source_url: str
    source_tier: int
    role: str = "supports"


@dataclass
class BlindSpot:
    description: str
    severity: str = "moderate"
    missing_source_family: str | None = None


@dataclass
class PatternCandidate:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    pattern_type: PatternType = PatternType.BRIDGE
    title: str = ""
    measured_pattern: str = ""
    evidence: list[EvidenceItem] = field(default_factory=list)
    counter_evidence: list[EvidenceItem] = field(default_factory=list)
    blind_spots: list[BlindSpot] = field(default_factory=list)
    confidence_score: float = 0.0
    confidence_level: ConfidenceLevel | None = None
    domain: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def evidence_count(self) -> int:
        return len(self.evidence)

    @property
    def source_tiers(self) -> set[int]:
        return {e.source_tier for e in self.evidence}

    @property
    def source_urls(self) -> set[str]:
        return {e.source_url for e in self.evidence}

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["pattern_type"] = self.pattern_type.value
        if self.confidence_level:
            d["confidence_level"] = self.confidence_level.value
        return d


@dataclass
class PromotedPattern(PatternCandidate):
    promoted_at: str = field(default_factory=lambda: datetime.now(tz=UTC).isoformat())
    promotion_reason: str = ""
    interpretation: str = ""
    category_integrity_passed: bool = True
    withheld_reason: str | None = None


@dataclass
class CoverageReport:
    source_families_used: list[str] = field(default_factory=list)
    source_families_missing: list[str] = field(default_factory=list)
    total_documents: int = 0
    documents_per_tier: dict[int, int] = field(default_factory=dict)
    weak_subtopics: list[str] = field(default_factory=list)
    sparse_graph_regions: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
