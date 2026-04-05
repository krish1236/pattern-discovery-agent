# Pattern Discovery Agent — Implementation Part 1
## Foundation: Types, Domain Packs, Connectors, Corpus, Extraction

**Read the full architecture in PATTERN_DISCOVERY_AGENT_ARCHITECTURE.md before starting.**

This doc covers Phases 1–5. Part 2 covers Phases 6–10 (graph, patterns, verifier, report, integration).

---

## Step 0: Project scaffold

Create the full directory structure and boilerplate files first.

### Directory structure

```
pattern-discovery-agent/
├── agent.py
├── agent.yaml
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── types.py
│   │   ├── graph.py              # Part 2
│   │   ├── patterns/
│   │   │   ├── __init__.py
│   │   │   ├── bridges.py        # Part 2
│   │   │   ├── contradictions.py # Part 2
│   │   │   ├── drift.py          # Part 2
│   │   │   └── gaps.py           # Part 2
│   │   ├── verifier.py           # Part 2
│   │   └── report.py             # Part 2
│   ├── domain_pack.py
│   ├── packs/
│   │   ├── __init__.py
│   │   └── research/
│   │       ├── __init__.py
│   │       ├── schema.py
│   │       ├── router.py
│   │       └── connectors/
│   │           ├── __init__.py
│   │           ├── openalex.py
│   │           ├── semantic_scholar.py
│   │           ├── arxiv.py
│   │           ├── github.py
│   │           └── web_search.py
│   └── shared/
│       ├── __init__.py
│       ├── corpus.py
│       ├── extraction.py
│       └── embeddings.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── core/
│   │   ├── __init__.py
│   │   └── test_types.py
│   ├── packs/
│   │   ├── __init__.py
│   │   └── research/
│   │       ├── __init__.py
│   │       ├── test_connectors.py
│   │       ├── test_schema.py
│   │       └── test_router.py
│   └── shared/
│       ├── __init__.py
│       ├── test_corpus.py
│       ├── test_extraction.py
│       └── test_embeddings.py
└── fixtures/
    ├── sample_openalex_response.json
    ├── sample_semantic_scholar_response.json
    ├── sample_arxiv_response.xml
    └── sample_assertions.json
```

### requirements.txt

```
agent-runtime>=0.1.0
httpx>=0.27.0
networkx>=3.3
sentence-transformers>=3.0.0
scikit-learn>=1.5.0
hdbscan>=0.8.38
numpy>=1.26.0
anthropic>=0.40.0
tenacity>=8.0.0
pytest>=8.0.0
pytest-asyncio>=0.23.0
```

### agent.yaml

```yaml
name: pattern-discovery
entrypoint: agent:run
python_version: "3.11"
browser: false
run_timeout_minutes: 60
max_concurrency: 2
tools: []
```

### All __init__.py files

All `__init__.py` files start empty. We add exports as we build each module.

---

## Phase 1: Universal type system

### File: src/core/types.py

This is the most important file in the project. Every other module depends on these types. No type may reference any domain-specific concept ("paper", "claim", "team", "bill"). All types must be fully domain-agnostic.

```python
"""
Universal type system for the Pattern Discovery Engine.

These types are domain-agnostic. The pattern engine, graph builder, verifier,
and report generator only ever see these types. Domain packs map their
domain-specific data (papers, game recaps, voting records) into these types.

RULE: No field, class name, or docstring in this file may reference
any specific domain (research, sports, politics). If you find yourself
writing "paper" or "claim", you're breaking the abstraction.
"""

from __future__ import annotations
import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, IntEnum
from typing import Any


class NodeType(str, Enum):
    """Universal node types. Every entity in the graph is one of these."""
    SOURCE_DOCUMENT = "source_document"
    ASSERTION = "assertion"
    ACTOR = "actor"
    CONCEPT = "concept"
    ARTIFACT = "artifact"
    METRIC = "metric"
    EVENT = "event"


class EdgeType(str, Enum):
    """Universal edge types between nodes."""
    ASSERTS = "asserts"            # SourceDocument -> Assertion
    SUPPORTS = "supports"          # Assertion -> Assertion
    CONTRADICTS = "contradicts"    # Assertion -> Assertion
    INVOLVES = "involves"          # Event -> Actor
    PRODUCES = "produces"          # Actor -> Artifact
    EVALUATES = "evaluates"        # Assertion -> Artifact
    ASSOCIATED_WITH = "associated_with"  # Actor -> Concept
    EXTENDS = "extends"            # Concept -> Concept
    CO_OCCURS = "co_occurs"        # Concept -> Concept
    PRECEDES = "precedes"          # Event -> Event
    BRIDGES_TO = "bridges_to"      # any -> any (added by pattern engine)
    RECOMMENDS = "recommends"      # Assertion -> Concept/Artifact (for gap detection)


class SourceTier(IntEnum):
    """Evidence quality tiers. Lower number = stronger evidence."""
    PRIMARY = 1           # Peer-reviewed papers, official stats, voting records
    STRONG_SECONDARY = 2  # Credible analysis, replications, expert commentary
    WEAK_SECONDARY = 3    # Vendor blogs, pundit predictions, opinion pieces
    SOCIAL = 4            # Forums, tweets, fan discussion, partisan commentary


class ConfidenceLevel(str, Enum):
    """Confidence labels for promoted patterns."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNRESOLVED = "unresolved"


class PatternType(str, Enum):
    """Types of patterns the engine can detect."""
    BRIDGE = "bridge"
    CONTRADICTION = "contradiction"
    DRIFT = "drift"
    GAP = "gap"


@dataclass
class EdgeMeta:
    """Metadata attached to every edge in the graph.
    
    This is how we track provenance. Every relationship in the graph
    carries its source, tier, confidence, and extraction details.
    """
    source_tier: int                # 1-4, maps to SourceTier
    source_family: str              # "scholarly", "code", "web", "official", "social"
    source_url: str                 # provenance URL
    source_id: str                  # API-specific ID
    extraction_confidence: float    # 0.0-1.0
    extraction_model: str           # "claude-haiku-4-5", "api_metadata", etc.
    timestamp: str                  # ISO date of the source
    provenance: str                 # "abstract", "full_text", "metadata", "api_field"
    domain: str                     # "research", "sports", "politics", etc.

    def __post_init__(self):
        if not 1 <= self.source_tier <= 4:
            raise ValueError(f"source_tier must be 1-4, got {self.source_tier}")
        if not 0.0 <= self.extraction_confidence <= 1.0:
            raise ValueError(f"extraction_confidence must be 0.0-1.0, got {self.extraction_confidence}")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> EdgeMeta:
        return cls(**d)


@dataclass
class Node:
    """A node in the knowledge graph.
    
    Every entity — regardless of domain — is represented as a Node
    with a universal NodeType and domain-specific metadata in the
    `properties` dict.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    node_type: NodeType = NodeType.CONCEPT
    name: str = ""
    description: str = ""
    domain: str = ""
    properties: dict[str, Any] = field(default_factory=dict)
    # Embedding vector — set by embedding pipeline, not during extraction
    embedding: list[float] | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["node_type"] = self.node_type.value
        # Don't serialize embeddings into JSON graph (too large)
        d.pop("embedding", None)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Node:
        d = d.copy()
        d["node_type"] = NodeType(d["node_type"])
        d.pop("embedding", None)
        return cls(**d)


@dataclass
class Edge:
    """An edge in the knowledge graph.
    
    Every relationship carries full provenance via EdgeMeta.
    """
    source_node_id: str
    target_node_id: str
    edge_type: EdgeType
    meta: EdgeMeta
    properties: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["edge_type"] = self.edge_type.value
        d["meta"] = self.meta.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Edge:
        d = d.copy()
        d["edge_type"] = EdgeType(d["edge_type"])
        d["meta"] = EdgeMeta.from_dict(d["meta"])
        return cls(**d)


@dataclass
class SourceDocument:
    """A document ingested from any source.
    
    This is the entry point for all data. Papers, articles, game recaps,
    bill texts — everything starts as a SourceDocument that gets
    converted into a Node + extracted into Assertions/Entities.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str = ""           # API-specific ID (OpenAlex ID, S2 corpus ID)
    source_family: str = ""       # "scholarly", "code", "web", etc.
    source_url: str = ""          # URL to the original
    title: str = ""
    abstract: str = ""            # or summary, description, etc.
    full_text: str | None = None
    authors: list[str] = field(default_factory=list)
    publication_date: str | None = None   # ISO date
    source_tier: int = 2          # default to Tier 2, domain pack overrides
    domain: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    # Pre-computed embedding from source API (e.g., SPECTER2 from S2)
    precomputed_embedding: list[float] | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("precomputed_embedding", None)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> SourceDocument:
        d = d.copy()
        d.pop("precomputed_embedding", None)
        return cls(**d)


@dataclass
class ExtractionResult:
    """Output from the LLM extraction pipeline for a single document.
    
    Contains all nodes and edges extracted from one SourceDocument.
    The extraction pipeline produces these; the graph builder consumes them.
    """
    source_document_id: str
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)


@dataclass
class EvidenceItem:
    """A single piece of evidence supporting or countering a pattern.
    
    Used in PatternCandidate to track what evidence backs each pattern.
    """
    assertion_node_id: str
    assertion_text: str
    source_document_id: str
    source_url: str
    source_tier: int
    role: str = "supports"     # "supports", "counters", "context"


@dataclass
class BlindSpot:
    """An explicitly identified gap in the evidence for a pattern."""
    description: str
    severity: str = "moderate"  # "minor", "moderate", "major"
    missing_source_family: str | None = None


@dataclass
class PatternCandidate:
    """A candidate pattern detected by the pattern engine.
    
    Must pass the promotion gate to become a PromotedPattern.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    pattern_type: PatternType = PatternType.BRIDGE
    title: str = ""
    measured_pattern: str = ""    # factual description of what the algorithm detected
    evidence: list[EvidenceItem] = field(default_factory=list)
    counter_evidence: list[EvidenceItem] = field(default_factory=list)
    blind_spots: list[BlindSpot] = field(default_factory=list)
    confidence_score: float = 0.0
    confidence_level: ConfidenceLevel | None = None
    domain: str = ""
    # Pattern-specific data (community IDs for bridges, time windows for drift, etc.)
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

    def to_dict(self) -> dict:
        d = asdict(self)
        d["pattern_type"] = self.pattern_type.value
        if self.confidence_level:
            d["confidence_level"] = self.confidence_level.value
        return d


@dataclass
class PromotedPattern(PatternCandidate):
    """A pattern that passed the promotion gate.
    
    Contains additional metadata from verification.
    """
    promoted_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    promotion_reason: str = ""
    interpretation: str = ""         # LLM synthesis of what the pattern means
    category_integrity_passed: bool = True
    withheld_reason: str | None = None  # set if demoted to exploratory


@dataclass
class CoverageReport:
    """Tracks what the agent searched and where it's weak."""
    source_families_used: list[str] = field(default_factory=list)
    source_families_missing: list[str] = field(default_factory=list)
    total_documents: int = 0
    documents_per_tier: dict[int, int] = field(default_factory=dict)
    weak_subtopics: list[str] = field(default_factory=list)
    sparse_graph_regions: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
```

### Test file: tests/core/test_types.py

```python
"""Tests for the universal type system.

Every test validates that types are domain-agnostic and serialize correctly.
"""
import json
import pytest
from src.core.types import (
    NodeType, EdgeType, SourceTier, ConfidenceLevel, PatternType,
    EdgeMeta, Node, Edge, SourceDocument, ExtractionResult,
    EvidenceItem, BlindSpot, PatternCandidate, PromotedPattern, CoverageReport,
)


class TestEdgeMeta:
    def test_valid_creation(self):
        meta = EdgeMeta(
            source_tier=1, source_family="scholarly", source_url="https://example.com",
            source_id="W123", extraction_confidence=0.9, extraction_model="claude-haiku-4-5",
            timestamp="2025-01-15", provenance="abstract", domain="research",
        )
        assert meta.source_tier == 1
        assert meta.extraction_confidence == 0.9

    def test_invalid_tier_raises(self):
        with pytest.raises(ValueError, match="source_tier must be 1-4"):
            EdgeMeta(source_tier=5, source_family="scholarly", source_url="",
                     source_id="", extraction_confidence=0.5, extraction_model="",
                     timestamp="", provenance="", domain="")

    def test_invalid_confidence_raises(self):
        with pytest.raises(ValueError, match="extraction_confidence must be 0.0-1.0"):
            EdgeMeta(source_tier=1, source_family="scholarly", source_url="",
                     source_id="", extraction_confidence=1.5, extraction_model="",
                     timestamp="", provenance="", domain="")

    def test_roundtrip_json(self):
        meta = EdgeMeta(
            source_tier=2, source_family="code", source_url="https://github.com/x",
            source_id="repo_123", extraction_confidence=0.85, extraction_model="api_metadata",
            timestamp="2025-06-01", provenance="metadata", domain="research",
        )
        d = meta.to_dict()
        restored = EdgeMeta.from_dict(d)
        assert restored.source_tier == meta.source_tier
        assert restored.source_url == meta.source_url
        assert restored.extraction_confidence == meta.extraction_confidence


class TestNode:
    def test_auto_generates_id(self):
        node = Node(node_type=NodeType.ASSERTION, name="test")
        assert len(node.id) == 36  # UUID format

    def test_roundtrip_json(self):
        node = Node(
            node_type=NodeType.ACTOR, name="Test Lab",
            description="A research lab", domain="research",
            properties={"affiliation": "MIT"},
        )
        d = node.to_dict()
        json_str = json.dumps(d)  # must be JSON-serializable
        restored = Node.from_dict(json.loads(json_str))
        assert restored.name == node.name
        assert restored.node_type == NodeType.ACTOR
        assert restored.properties["affiliation"] == "MIT"

    def test_embedding_not_serialized(self):
        node = Node(node_type=NodeType.CONCEPT, name="attention",
                    embedding=[0.1, 0.2, 0.3])
        d = node.to_dict()
        assert "embedding" not in d


class TestEdge:
    def test_roundtrip_json(self):
        meta = EdgeMeta(
            source_tier=1, source_family="scholarly", source_url="https://example.com",
            source_id="W123", extraction_confidence=0.9, extraction_model="claude-haiku-4-5",
            timestamp="2025-01-15", provenance="abstract", domain="research",
        )
        edge = Edge(
            source_node_id="node_1", target_node_id="node_2",
            edge_type=EdgeType.SUPPORTS, meta=meta,
        )
        d = edge.to_dict()
        json_str = json.dumps(d)
        restored = Edge.from_dict(json.loads(json_str))
        assert restored.edge_type == EdgeType.SUPPORTS
        assert restored.meta.source_tier == 1


class TestSourceDocument:
    def test_roundtrip_json(self):
        doc = SourceDocument(
            source_id="W2741809807", source_family="scholarly",
            source_url="https://openalex.org/W2741809807",
            title="Attention Is All You Need", abstract="We propose...",
            authors=["Vaswani, A."], publication_date="2017-06-12",
            source_tier=1, domain="research",
            metadata={"citation_count": 90000},
        )
        d = doc.to_dict()
        json_str = json.dumps(d)
        restored = SourceDocument.from_dict(json.loads(json_str))
        assert restored.title == doc.title
        assert restored.source_tier == 1

    def test_precomputed_embedding_not_serialized(self):
        doc = SourceDocument(title="test", precomputed_embedding=[0.1, 0.2])
        d = doc.to_dict()
        assert "precomputed_embedding" not in d


class TestPatternCandidate:
    def test_evidence_count(self):
        evidence = [
            EvidenceItem(assertion_node_id="a1", assertion_text="claim 1",
                        source_document_id="d1", source_url="https://a.com", source_tier=1),
            EvidenceItem(assertion_node_id="a2", assertion_text="claim 2",
                        source_document_id="d2", source_url="https://b.com", source_tier=2),
        ]
        pattern = PatternCandidate(
            pattern_type=PatternType.BRIDGE, title="Test bridge",
            evidence=evidence,
        )
        assert pattern.evidence_count == 2
        assert pattern.source_tiers == {1, 2}
        assert len(pattern.source_urls) == 2

    def test_serialization(self):
        pattern = PatternCandidate(
            pattern_type=PatternType.CONTRADICTION,
            title="Test contradiction",
            confidence_level=ConfidenceLevel.MEDIUM,
            domain="research",
        )
        d = pattern.to_dict()
        assert d["pattern_type"] == "contradiction"
        assert d["confidence_level"] == "medium"


class TestDomainAgnosticism:
    """Verify no domain-specific language leaked into types."""
    
    def test_node_types_are_generic(self):
        for nt in NodeType:
            assert "paper" not in nt.value
            assert "claim" not in nt.value
            assert "team" not in nt.value
            assert "bill" not in nt.value

    def test_edge_types_are_generic(self):
        for et in EdgeType:
            assert "cites" not in et.value  # domain-specific
            assert "authored" not in et.value
```

**Run tests:** `pytest tests/core/test_types.py -v`

**Done when:** All tests pass. Types are clean, serializable, domain-agnostic.

---

## Phase 2: Domain pack interface + research pack schema

### File: src/domain_pack.py

```python
"""
Domain pack base class and interfaces.

A domain pack provides three things:
1. Source connectors — which APIs to call
2. Schema — how to map domain data to universal types
3. Interpretation — how to explain patterns in domain language

The core engine never imports a specific domain pack directly.
It receives one via configuration and calls the interface methods.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.types import SourceDocument, PatternCandidate


class SourceConnector(ABC):
    """Base class for all source connectors.
    
    Each connector wraps one external API and returns SourceDocuments.
    """
    
    @abstractmethod
    async def search(self, query: str, limit: int = 20, **kwargs) -> list[SourceDocument]:
        """Search the source for documents matching the query."""
        ...

    @abstractmethod
    async def get(self, source_id: str) -> SourceDocument | None:
        """Fetch a single document by its source-specific ID."""
        ...

    @abstractmethod
    async def expand(self, source_id: str, limit: int = 10) -> list[SourceDocument]:
        """Find related documents (citations, similar, linked)."""
        ...


@dataclass
class DomainSchema:
    """Schema mapping domain-specific concepts to universal types.
    
    The extraction pipeline uses this to know what to extract
    and how to map it to universal node/edge types.
    """
    domain: str                                  # "research", "sports", "politics"
    entity_types: dict[str, str]                 # domain label -> NodeType value
    edge_types: dict[str, str]                   # domain label -> EdgeType value
    tier_rules: dict[str, int]                   # source classifier -> tier number
    extraction_prompt: str                       # LLM prompt for structured extraction
    interpretation_templates: dict[str, str]      # pattern_type -> template string


class DomainPack(ABC):
    """Base class for domain packs.
    
    Subclass this for each domain (research, sports, politics).
    The core engine calls these methods — it never writes
    domain-specific logic itself.
    """

    @property
    @abstractmethod
    def domain(self) -> str:
        """Domain identifier: 'research', 'sports', 'politics', etc."""
        ...

    @abstractmethod
    def get_schema(self) -> DomainSchema:
        """Return the domain schema with entity types, tier rules, prompts."""
        ...

    @abstractmethod
    def get_connectors(self, config: dict[str, Any]) -> list[SourceConnector]:
        """Return configured source connectors for this domain.
        
        Args:
            config: dict with API keys and settings from env vars.
        """
        ...

    @abstractmethod
    def classify_tier(self, doc: SourceDocument) -> int:
        """Assign a source tier (1-4) to a document based on domain rules."""
        ...

    @abstractmethod
    def interpret(self, pattern: PatternCandidate) -> str:
        """Generate a domain-appropriate interpretation of a pattern.
        
        Returns a prompt fragment for the LLM to use when synthesizing
        the pattern explanation in the report.
        """
        ...
```

### File: src/packs/research/schema.py

```python
"""
Research domain pack schema.

Maps research-specific concepts (papers, claims, methods, authors)
to the universal type system. Provides extraction prompts and
interpretation templates specific to research/technical topics.
"""

from src.domain_pack import DomainSchema


RESEARCH_EXTRACTION_PROMPT = """You are extracting structured knowledge from a research document.

Given the following text (title + abstract), extract:

1. ASSERTIONS: Verifiable claims the document makes.
   Each assertion needs: text, conditions (if any), polarity (positive/negative/neutral).

2. ENTITIES: Named things mentioned.
   Each entity needs: name, entity_type (one of: actor, concept, artifact, metric).
   - actor: people, labs, organizations, companies
   - concept: methods, techniques, ideas, approaches, algorithms
   - artifact: datasets, benchmarks, tools, repositories, models
   - metric: measurable quantities (accuracy, F1, latency, cost)

3. RELATIONSHIPS: How entities and assertions connect.
   Each relationship needs: source_name, target_name, relationship_type.
   Allowed relationship types: supports, contradicts, extends, evaluates,
   produces, associated_with, co_occurs, recommends.

Respond ONLY with valid JSON in this exact format:
{
  "assertions": [
    {"text": "...", "conditions": "...", "polarity": "positive"}
  ],
  "entities": [
    {"name": "...", "entity_type": "concept", "description": "..."}
  ],
  "relationships": [
    {"source_name": "...", "target_name": "...", "relationship_type": "supports"}
  ]
}

Do NOT include any text outside the JSON. Do NOT wrap in markdown code blocks.
If the text has no extractable content, return {"assertions": [], "entities": [], "relationships": []}.

Text to extract from:
---
Title: {title}
Abstract: {abstract}
---"""


RESEARCH_INTERPRETATION_TEMPLATES = {
    "bridge": (
        "Two research communities ({community_a} and {community_b}) are working on "
        "related problems using different terminology and rarely citing each other. "
        "The connecting thread is {bridge_entity}. "
        "Explain why this bridge matters and what each community could learn from the other."
    ),
    "contradiction": (
        "Source A claims: \"{assertion_a}\" (from {source_a}). "
        "Source B claims: \"{assertion_b}\" (from {source_b}). "
        "These assertions appear to contradict. Determine: "
        "1) Is this a real contradiction or an apparent one (different conditions, definitions, or populations)? "
        "2) What conditions make each claim true? "
        "3) What additional evidence would resolve this?"
    ),
    "drift": (
        "This research area has evolved across these time windows: {time_windows}. "
        "The following cluster transitions were detected: {transitions}. "
        "Describe the phase shift: what was the field doing before, what changed, and what is emerging now."
    ),
    "gap": (
        "The following concept/method/experiment is frequently recommended across {recommender_count} sources "
        "but has {execution_count} actual results in the literature: {gap_description}. "
        "Explain why this gap likely exists and what barriers prevent execution."
    ),
}


RESEARCH_TIER_RULES = {
    "peer_reviewed": 1,
    "official_benchmark": 1,
    "official_documentation": 1,
    "first_party_report": 1,
    "arxiv_preprint": 2,
    "credible_analysis": 2,
    "implementation_writeup": 2,
    "benchmark_replication": 2,
    "vendor_blog": 3,
    "newsletter": 3,
    "general_article": 3,
    "forum_post": 4,
    "social_media": 4,
    "unverified_commentary": 4,
}


RESEARCH_ENTITY_TYPES = {
    "paper": "source_document",
    "claim": "assertion",
    "author": "actor",
    "lab": "actor",
    "organization": "actor",
    "method": "concept",
    "technique": "concept",
    "algorithm": "concept",
    "dataset": "artifact",
    "benchmark": "artifact",
    "model": "artifact",
    "repository": "artifact",
    "tool": "artifact",
    "citation_count": "metric",
    "accuracy": "metric",
    "f1_score": "metric",
    "publication": "event",
    "benchmark_release": "event",
}


RESEARCH_EDGE_TYPES = {
    "cites": "asserts",
    "claims": "asserts",
    "authored_by": "involves",
    "evaluated_on": "evaluates",
    "implemented_in": "produces",
    "builds_on": "extends",
    "related_to": "co_occurs",
    "suggests_future_work": "recommends",
}


def get_research_schema() -> DomainSchema:
    return DomainSchema(
        domain="research",
        entity_types=RESEARCH_ENTITY_TYPES,
        edge_types=RESEARCH_EDGE_TYPES,
        tier_rules=RESEARCH_TIER_RULES,
        extraction_prompt=RESEARCH_EXTRACTION_PROMPT,
        interpretation_templates=RESEARCH_INTERPRETATION_TEMPLATES,
    )
```

### File: src/packs/research/router.py

```python
"""
Research domain router.

Classifies whether a topic belongs to the research/technical domain
and builds a source plan (which connectors to use, with what budgets).
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class SourcePlan:
    """Plan for which sources to query and how deeply."""
    connectors: list[str]              # connector names to activate
    queries: list[str]                 # search queries to run
    max_documents: int = 100           # total document budget
    expansion_budget: int = 50         # how many docs to add via expansion
    per_connector_limit: int = 30      # max docs per connector in initial search
    time_filter: str | None = None     # ISO date range filter


RESEARCH_KEYWORDS = [
    "research", "paper", "study", "algorithm", "framework", "model",
    "benchmark", "dataset", "architecture", "neural", "transformer",
    "machine learning", "deep learning", "NLP", "computer vision",
    "reinforcement learning", "optimization", "distributed systems",
    "database", "compiler", "operating system", "network protocol",
    "security", "cryptography", "software engineering", "API",
    "microservice", "kubernetes", "cloud", "serverless", "edge computing",
    "LLM", "GPT", "BERT", "diffusion", "generative", "agent",
    "retrieval", "RAG", "knowledge graph", "embedding", "fine-tuning",
    "evaluation", "scaling", "inference", "training", "pretraining",
]


def is_research_topic(topic: str) -> bool:
    """Check if a topic string looks like a research/technical topic.
    
    Simple keyword match for now. The domain classifier in the main
    pipeline will use an LLM for ambiguous cases. This is the fast path.
    """
    topic_lower = topic.lower()
    return any(kw in topic_lower for kw in RESEARCH_KEYWORDS)


def build_source_plan(
    topic: str,
    depth: str = "standard",
    focus: str = "all",
    time_range: str | None = None,
    max_documents: int = 100,
) -> SourcePlan:
    """Build a source plan for a research topic.
    
    Args:
        topic: The research topic to investigate
        depth: "quick" (50 docs), "standard" (100), "deep" (200)
        focus: "bridges", "contradictions", "drift", "gaps", or "all"
        time_range: Optional ISO date range like "2023-2026"
        max_documents: Override for max document count
    """
    depth_limits = {"quick": 50, "standard": 100, "deep": 200}
    max_docs = depth_limits.get(depth, max_documents)
    
    # Always use OpenAlex and Semantic Scholar as primary
    connectors = ["openalex", "semantic_scholar"]
    
    # Add arXiv for preprints
    connectors.append("arxiv")
    
    # Add GitHub for implementation-heavy topics
    code_keywords = ["framework", "library", "tool", "sdk", "api", "implementation"]
    if any(kw in topic.lower() for kw in code_keywords):
        connectors.append("github")
    
    # Add web search for broader context
    connectors.append("web_search")
    
    # Build search queries: main topic + subtopic variations
    queries = [topic]
    # Add the topic with "survey" and "benchmark" for comprehensive coverage
    queries.append(f"{topic} survey")
    queries.append(f"{topic} benchmark evaluation")
    
    # For drift focus, add temporal queries
    if focus in ("drift", "all"):
        queries.append(f"{topic} recent advances")
        queries.append(f"{topic} emerging trends")
    
    per_connector = max(10, max_docs // len(connectors))
    expansion_budget = max(20, max_docs // 3)
    
    return SourcePlan(
        connectors=connectors,
        queries=queries,
        max_documents=max_docs,
        expansion_budget=expansion_budget,
        per_connector_limit=per_connector,
        time_filter=time_range,
    )
```

### File: src/packs/research/__init__.py

```python
"""Research domain pack — first domain implementation."""

from __future__ import annotations
from typing import Any

from src.core.types import SourceDocument, PatternCandidate, PatternType
from src.domain_pack import DomainPack, DomainSchema, SourceConnector
from src.packs.research.schema import get_research_schema
from src.packs.research.router import build_source_plan, is_research_topic, SourcePlan


class ResearchPack(DomainPack):
    """Research/Technical domain pack.
    
    Covers: AI/ML, software engineering, systems, scientific literature.
    Sources: OpenAlex, Semantic Scholar, arXiv, GitHub, web search.
    """

    @property
    def domain(self) -> str:
        return "research"

    def get_schema(self) -> DomainSchema:
        return get_research_schema()

    def get_connectors(self, config: dict[str, Any]) -> list[SourceConnector]:
        # Import connectors here to avoid circular imports
        from src.packs.research.connectors.openalex import OpenAlexConnector
        from src.packs.research.connectors.semantic_scholar import SemanticScholarConnector
        from src.packs.research.connectors.arxiv import ArxivConnector
        from src.packs.research.connectors.github import GitHubConnector
        from src.packs.research.connectors.web_search import WebSearchConnector

        connectors = [
            OpenAlexConnector(api_key=config.get("OPENALEX_API_KEY", "")),
            SemanticScholarConnector(api_key=config.get("SEMANTIC_SCHOLAR_API_KEY", "")),
            ArxivConnector(),
        ]
        if config.get("GITHUB_TOKEN"):
            connectors.append(GitHubConnector(token=config["GITHUB_TOKEN"]))
        if config.get("TAVILY_API_KEY"):
            connectors.append(WebSearchConnector(api_key=config["TAVILY_API_KEY"]))
        return connectors

    def classify_tier(self, doc: SourceDocument) -> int:
        """Classify source tier based on research-specific rules."""
        schema = self.get_schema()
        
        # Check metadata for peer review status
        is_peer_reviewed = doc.metadata.get("is_peer_reviewed", False)
        source_type = doc.metadata.get("source_type", "")
        
        if is_peer_reviewed or source_type == "journal":
            return 1
        if doc.source_family == "scholarly":
            return 2  # arXiv preprints, conference papers without explicit peer review flag
        if doc.source_family == "code":
            return 2  # GitHub repos with real implementations
        if doc.source_family == "web":
            return 3  # blogs, docs
        if doc.source_family == "social":
            return 4
        return 3  # default to weak secondary

    def interpret(self, pattern: PatternCandidate) -> str:
        """Return an interpretation prompt fragment for the pattern."""
        schema = self.get_schema()
        template = schema.interpretation_templates.get(pattern.pattern_type.value, "")
        # Template variables are filled by the report generator using pattern.details
        return template
```

### Test file: tests/packs/research/test_schema.py

```python
"""Tests for research domain pack."""
import pytest
from src.core.types import NodeType, EdgeType, SourceDocument, PatternCandidate, PatternType
from src.packs.research import ResearchPack
from src.packs.research.schema import get_research_schema
from src.packs.research.router import is_research_topic, build_source_plan


class TestResearchSchema:
    def test_entity_types_map_to_valid_node_types(self):
        schema = get_research_schema()
        valid_values = {nt.value for nt in NodeType}
        for domain_label, universal_type in schema.entity_types.items():
            assert universal_type in valid_values, f"{domain_label} maps to invalid type {universal_type}"

    def test_edge_types_map_to_valid_edge_types(self):
        schema = get_research_schema()
        valid_values = {et.value for et in EdgeType}
        for domain_label, universal_type in schema.edge_types.items():
            assert universal_type in valid_values, f"{domain_label} maps to invalid type {universal_type}"

    def test_extraction_prompt_contains_required_placeholders(self):
        schema = get_research_schema()
        assert "{title}" in schema.extraction_prompt
        assert "{abstract}" in schema.extraction_prompt

    def test_interpretation_templates_exist_for_all_pattern_types(self):
        schema = get_research_schema()
        for pt in PatternType:
            assert pt.value in schema.interpretation_templates, f"Missing template for {pt.value}"

    def test_tier_rules_cover_key_source_types(self):
        schema = get_research_schema()
        assert schema.tier_rules["peer_reviewed"] == 1
        assert schema.tier_rules["arxiv_preprint"] == 2
        assert schema.tier_rules["vendor_blog"] == 3
        assert schema.tier_rules["forum_post"] == 4


class TestResearchRouter:
    def test_classifies_research_topics(self):
        assert is_research_topic("transformer architectures for NLP") is True
        assert is_research_topic("reinforcement learning in robotics") is True
        assert is_research_topic("AI agent frameworks 2024-2026") is True

    def test_source_plan_includes_primary_connectors(self):
        plan = build_source_plan("transformer architectures")
        assert "openalex" in plan.connectors
        assert "semantic_scholar" in plan.connectors

    def test_source_plan_depth_affects_limits(self):
        quick = build_source_plan("test", depth="quick")
        deep = build_source_plan("test", depth="deep")
        assert quick.max_documents < deep.max_documents

    def test_source_plan_adds_github_for_code_topics(self):
        plan = build_source_plan("Python web framework comparison")
        assert "github" in plan.connectors

    def test_source_plan_generates_multiple_queries(self):
        plan = build_source_plan("AI agents")
        assert len(plan.queries) >= 2  # at least main topic + survey variant


class TestResearchPack:
    def test_domain_is_research(self):
        pack = ResearchPack()
        assert pack.domain == "research"

    def test_classify_tier_peer_reviewed(self):
        pack = ResearchPack()
        doc = SourceDocument(
            title="Test", source_family="scholarly",
            metadata={"is_peer_reviewed": True},
        )
        assert pack.classify_tier(doc) == 1

    def test_classify_tier_arxiv(self):
        pack = ResearchPack()
        doc = SourceDocument(title="Test", source_family="scholarly")
        assert pack.classify_tier(doc) == 2

    def test_classify_tier_blog(self):
        pack = ResearchPack()
        doc = SourceDocument(title="Test", source_family="web")
        assert pack.classify_tier(doc) == 3

    def test_interpret_returns_template(self):
        pack = ResearchPack()
        pattern = PatternCandidate(pattern_type=PatternType.BRIDGE)
        result = pack.interpret(pattern)
        assert len(result) > 0
        assert "communities" in result.lower() or "community" in result.lower()
```

**Run tests:** `pytest tests/core/test_types.py tests/packs/research/ -v`

**Done when:** All type tests pass. Research schema maps correctly to universal types. Router classifies topics and builds source plans.

---

## Phase 3: Source connectors

### File: src/packs/research/connectors/openalex.py

```python
"""
OpenAlex source connector.

API docs: https://docs.openalex.org
Rate limits: Free key = 100K credits/day. Singleton requests = free.
List requests = 1 credit each. Max 100 requests/second.

Key features used:
- /works?search= for keyword search
- /works?search= with semantic search (paste abstract to find similar)
- /works/{id}/cited_by for citation expansion
- Fields: title, abstract_inverted_index, publication_date, cited_by_count,
  authorships, primary_location, type, is_oa, concepts
"""

from __future__ import annotations
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.core.types import SourceDocument
from src.domain_pack import SourceConnector

OPENALEX_BASE = "https://api.openalex.org"


def _invert_abstract(inverted: dict | None) -> str:
    """Convert OpenAlex inverted abstract index to plain text."""
    if not inverted:
        return ""
    word_positions = []
    for word, positions in inverted.items():
        for pos in positions:
            word_positions.append((pos, word))
    word_positions.sort()
    return " ".join(word for _, word in word_positions)


def _parse_work(work: dict, domain: str = "research") -> SourceDocument:
    """Convert an OpenAlex work object to a SourceDocument."""
    openalex_id = work.get("id", "").replace("https://openalex.org/", "")
    
    # Determine source tier from type
    work_type = work.get("type", "")
    is_peer_reviewed = work.get("primary_location", {}).get("is_oa", False)
    source_type = ""
    if work_type in ("journal-article", "proceedings-article"):
        source_type = "journal"
    
    authors = []
    for authorship in work.get("authorships", [])[:10]:  # cap at 10
        author_name = authorship.get("author", {}).get("display_name", "")
        if author_name:
            authors.append(author_name)
    
    abstract = _invert_abstract(work.get("abstract_inverted_index"))
    
    return SourceDocument(
        source_id=openalex_id,
        source_family="scholarly",
        source_url=work.get("doi", "") or work.get("id", ""),
        title=work.get("title", "") or "",
        abstract=abstract,
        authors=authors,
        publication_date=work.get("publication_date"),
        source_tier=1 if source_type == "journal" else 2,
        domain=domain,
        metadata={
            "openalex_id": openalex_id,
            "cited_by_count": work.get("cited_by_count", 0),
            "type": work_type,
            "source_type": source_type,
            "is_peer_reviewed": source_type == "journal",
            "concepts": [c.get("display_name", "") for c in work.get("concepts", [])[:5]],
            "is_oa": work.get("is_oa", False),
        },
    )


class OpenAlexConnector(SourceConnector):
    """OpenAlex API connector for scholarly works."""

    def __init__(self, api_key: str = "", email: str = ""):
        self.api_key = api_key
        self.email = email
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            if self.email:
                headers["User-Agent"] = f"mailto:{self.email}"
            self._client = httpx.AsyncClient(
                base_url=OPENALEX_BASE,
                headers=headers,
                timeout=30.0,
            )
        return self._client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.ConnectError)),
    )
    async def search(self, query: str, limit: int = 20, **kwargs) -> list[SourceDocument]:
        """Search OpenAlex works by keyword."""
        client = await self._get_client()
        params = {
            "search": query,
            "per_page": min(limit, 50),
            "sort": "cited_by_count:desc",
        }
        # Add date filter if provided
        time_filter = kwargs.get("time_filter")
        if time_filter and "-" in time_filter:
            start, end = time_filter.split("-", 1)
            params["filter"] = f"publication_year:{start.strip()}-{end.strip()}"
        
        resp = await client.get("/works", params=params)
        resp.raise_for_status()
        data = resp.json()
        
        results = []
        for work in data.get("results", []):
            doc = _parse_work(work)
            if doc.title:  # skip works without titles
                results.append(doc)
        return results

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.ConnectError)),
    )
    async def get(self, source_id: str) -> SourceDocument | None:
        """Fetch a single work by OpenAlex ID."""
        client = await self._get_client()
        try:
            resp = await client.get(f"/works/{source_id}")
            resp.raise_for_status()
            return _parse_work(resp.json())
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.ConnectError)),
    )
    async def expand(self, source_id: str, limit: int = 10) -> list[SourceDocument]:
        """Find papers that cite the given work (citation expansion)."""
        client = await self._get_client()
        params = {
            "filter": f"cites:{source_id}",
            "per_page": min(limit, 50),
            "sort": "cited_by_count:desc",
        }
        resp = await client.get("/works", params=params)
        resp.raise_for_status()
        data = resp.json()
        
        results = []
        for work in data.get("results", []):
            doc = _parse_work(work)
            if doc.title:
                results.append(doc)
        return results

    async def semantic_search(self, text: str, limit: int = 10) -> list[SourceDocument]:
        """Find semantically similar papers using OpenAlex's vector search.
        
        This is crucial for bridge detection — finds papers that discuss
        the same mechanisms using different terminology.
        """
        client = await self._get_client()
        params = {
            "search": text[:500],  # cap query length
            "per_page": min(limit, 50),
        }
        resp = await client.get("/works", params=params)
        resp.raise_for_status()
        data = resp.json()
        
        results = []
        for work in data.get("results", []):
            doc = _parse_work(work)
            if doc.title:
                results.append(doc)
        return results

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None
```

### File: src/packs/research/connectors/semantic_scholar.py

```python
"""
Semantic Scholar source connector.

API docs: https://api.semanticscholar.org/api-docs/
Rate limits: Free key = 1 RPS. Unauthenticated = shared pool (unreliable).
Batch endpoint: POST /paper/batch with up to 500 IDs.

Key features used:
- /paper/search for keyword search
- /paper/batch for bulk metadata fetch
- /paper/{id}/citations for cited-by
- /paper/{id}/references for references
- Fields: title, abstract, year, citationCount, influentialCitationCount,
  tldr, embedding (SPECTER2), authors, externalIds
"""

from __future__ import annotations
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.core.types import SourceDocument
from src.domain_pack import SourceConnector

S2_BASE = "https://api.semanticscholar.org/graph/v1"

FIELDS = "title,abstract,year,citationCount,influentialCitationCount,tldr,authors,externalIds,publicationDate,embedding.specter_v2"


def _parse_paper(paper: dict, domain: str = "research") -> SourceDocument:
    """Convert an S2 paper object to a SourceDocument."""
    s2_id = paper.get("paperId", "")
    
    authors = []
    for author in paper.get("authors", [])[:10]:
        name = author.get("name", "")
        if name:
            authors.append(name)
    
    # Extract SPECTER2 embedding if available
    embedding = None
    emb_data = paper.get("embedding")
    if emb_data and isinstance(emb_data, dict):
        embedding = emb_data.get("vector")
    
    # Get DOI for cross-referencing with OpenAlex
    external_ids = paper.get("externalIds") or {}
    doi = external_ids.get("DOI", "")
    doi_url = f"https://doi.org/{doi}" if doi else ""
    
    # TLDR for quick summary
    tldr = ""
    tldr_data = paper.get("tldr")
    if tldr_data and isinstance(tldr_data, dict):
        tldr = tldr_data.get("text", "")
    
    return SourceDocument(
        source_id=s2_id,
        source_family="scholarly",
        source_url=doi_url or f"https://www.semanticscholar.org/paper/{s2_id}",
        title=paper.get("title", "") or "",
        abstract=paper.get("abstract", "") or "",
        authors=authors,
        publication_date=paper.get("publicationDate"),
        source_tier=2,  # default; pack.classify_tier() refines this
        domain=domain,
        metadata={
            "s2_id": s2_id,
            "doi": doi,
            "citation_count": paper.get("citationCount", 0),
            "influential_citation_count": paper.get("influentialCitationCount", 0),
            "year": paper.get("year"),
            "tldr": tldr,
        },
        precomputed_embedding=embedding,
    )


class SemanticScholarConnector(SourceConnector):
    """Semantic Scholar Academic Graph API connector."""

    def __init__(self, api_key: str = ""):
        self.api_key = api_key
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            headers = {}
            if self.api_key:
                headers["x-api-key"] = self.api_key
            self._client = httpx.AsyncClient(
                base_url=S2_BASE,
                headers=headers,
                timeout=30.0,
            )
        return self._client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=3, max=15),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.ConnectError)),
    )
    async def search(self, query: str, limit: int = 20, **kwargs) -> list[SourceDocument]:
        client = await self._get_client()
        params = {
            "query": query,
            "limit": min(limit, 100),
            "fields": FIELDS,
        }
        year_filter = kwargs.get("time_filter")
        if year_filter and "-" in year_filter:
            start, end = year_filter.split("-", 1)
            params["year"] = f"{start.strip()}-{end.strip()}"
        
        resp = await client.get("/paper/search", params=params)
        resp.raise_for_status()
        data = resp.json()
        
        results = []
        for paper in data.get("data", []):
            doc = _parse_paper(paper)
            if doc.title:
                results.append(doc)
        return results

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=3, max=15),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.ConnectError)),
    )
    async def get(self, source_id: str) -> SourceDocument | None:
        client = await self._get_client()
        try:
            resp = await client.get(f"/paper/{source_id}", params={"fields": FIELDS})
            resp.raise_for_status()
            return _parse_paper(resp.json())
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=3, max=15),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.ConnectError)),
    )
    async def expand(self, source_id: str, limit: int = 10) -> list[SourceDocument]:
        """Get papers that cite this paper."""
        client = await self._get_client()
        resp = await client.get(
            f"/paper/{source_id}/citations",
            params={"fields": "title,abstract,year,citationCount,authors,externalIds,publicationDate", "limit": min(limit, 100)},
        )
        resp.raise_for_status()
        data = resp.json()
        
        results = []
        for item in data.get("data", []):
            citing_paper = item.get("citingPaper", {})
            if citing_paper and citing_paper.get("title"):
                doc = _parse_paper(citing_paper)
                results.append(doc)
        return results

    async def batch_get(self, paper_ids: list[str]) -> list[SourceDocument]:
        """Fetch multiple papers in a single batch request."""
        client = await self._get_client()
        resp = await client.post(
            "/paper/batch",
            params={"fields": FIELDS},
            json={"ids": paper_ids[:500]},  # API limit
        )
        resp.raise_for_status()
        
        results = []
        for paper in resp.json():
            if paper and paper.get("title"):
                results.append(_parse_paper(paper))
        return results

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None
```

### File: src/packs/research/connectors/arxiv.py

```python
"""
arXiv source connector.

API docs: https://info.arxiv.org/help/api/user-manual.html
Rate limits: No key required. Please be polite (3 second delay between requests).
Returns: Atom XML feed.

Used for: full abstracts, preprint access, CS/ML/physics papers.
"""

from __future__ import annotations
import asyncio
import xml.etree.ElementTree as ET
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.core.types import SourceDocument
from src.domain_pack import SourceConnector

ARXIV_BASE = "http://export.arxiv.org/api"
NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


def _parse_entry(entry: ET.Element, domain: str = "research") -> SourceDocument:
    """Convert an arXiv Atom entry to a SourceDocument."""
    title = (entry.findtext("atom:title", "", NS) or "").strip().replace("\n", " ")
    abstract = (entry.findtext("atom:summary", "", NS) or "").strip().replace("\n", " ")
    
    arxiv_id = (entry.findtext("atom:id", "", NS) or "").split("/abs/")[-1]
    published = entry.findtext("atom:published", "", NS)[:10] if entry.findtext("atom:published", "", NS) else None
    
    authors = []
    for author in entry.findall("atom:author", NS)[:10]:
        name = author.findtext("atom:name", "", NS)
        if name:
            authors.append(name)
    
    categories = []
    for cat in entry.findall("arxiv:primary_category", NS):
        term = cat.get("term", "")
        if term:
            categories.append(term)
    
    return SourceDocument(
        source_id=arxiv_id,
        source_family="scholarly",
        source_url=f"https://arxiv.org/abs/{arxiv_id}",
        title=title,
        abstract=abstract,
        authors=authors,
        publication_date=published,
        source_tier=2,  # arXiv preprints = Tier 2
        domain=domain,
        metadata={
            "arxiv_id": arxiv_id,
            "categories": categories,
            "source_type": "preprint",
        },
    )


class ArxivConnector(SourceConnector):
    """arXiv API connector."""

    def __init__(self):
        self._client: httpx.AsyncClient | None = None
        self._last_request_time: float = 0

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def _rate_limit(self):
        """arXiv asks for 3-second delay between requests."""
        import time
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < 3.0:
            await asyncio.sleep(3.0 - elapsed)
        self._last_request_time = time.time()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=5, max=30),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.ConnectError)),
    )
    async def search(self, query: str, limit: int = 20, **kwargs) -> list[SourceDocument]:
        await self._rate_limit()
        client = await self._get_client()
        
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": min(limit, 50),
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        
        resp = await client.get(f"{ARXIV_BASE}/query", params=params)
        resp.raise_for_status()
        
        root = ET.fromstring(resp.text)
        results = []
        for entry in root.findall("atom:entry", NS):
            doc = _parse_entry(entry)
            if doc.title and doc.title != "Error":
                results.append(doc)
        return results

    async def get(self, source_id: str) -> SourceDocument | None:
        await self._rate_limit()
        client = await self._get_client()
        
        params = {"id_list": source_id, "max_results": 1}
        resp = await client.get(f"{ARXIV_BASE}/query", params=params)
        resp.raise_for_status()
        
        root = ET.fromstring(resp.text)
        entries = root.findall("atom:entry", NS)
        if entries:
            doc = _parse_entry(entries[0])
            if doc.title and doc.title != "Error":
                return doc
        return None

    async def expand(self, source_id: str, limit: int = 10) -> list[SourceDocument]:
        """arXiv doesn't support citation expansion. Return empty."""
        return []

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None
```

### File: src/packs/research/connectors/github.py

```python
"""
GitHub source connector.

API docs: https://docs.github.com/en/rest
Rate limits: 60/hour unauthenticated, 5000/hour with token.

Used for: finding implementations, popular repos related to the topic.
"""

from __future__ import annotations
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.core.types import SourceDocument
from src.domain_pack import SourceConnector

GITHUB_API = "https://api.github.com"


def _parse_repo(repo: dict, domain: str = "research") -> SourceDocument:
    return SourceDocument(
        source_id=str(repo.get("id", "")),
        source_family="code",
        source_url=repo.get("html_url", ""),
        title=repo.get("full_name", ""),
        abstract=repo.get("description", "") or "",
        authors=[repo.get("owner", {}).get("login", "")],
        publication_date=repo.get("created_at", "")[:10] if repo.get("created_at") else None,
        source_tier=2,  # real implementations = Tier 2
        domain=domain,
        metadata={
            "stars": repo.get("stargazers_count", 0),
            "language": repo.get("language", ""),
            "forks": repo.get("forks_count", 0),
            "topics": repo.get("topics", []),
            "updated_at": repo.get("updated_at"),
            "source_type": "repository",
        },
    )


class GitHubConnector(SourceConnector):
    def __init__(self, token: str = ""):
        self.token = token
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            headers = {"Accept": "application/vnd.github+json"}
            if self.token:
                headers["Authorization"] = f"Bearer {self.token}"
            self._client = httpx.AsyncClient(
                base_url=GITHUB_API, headers=headers, timeout=30.0,
            )
        return self._client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.ConnectError)),
    )
    async def search(self, query: str, limit: int = 10, **kwargs) -> list[SourceDocument]:
        client = await self._get_client()
        params = {
            "q": query,
            "sort": "stars",
            "order": "desc",
            "per_page": min(limit, 30),
        }
        resp = await client.get("/search/repositories", params=params)
        resp.raise_for_status()
        
        results = []
        for repo in resp.json().get("items", []):
            doc = _parse_repo(repo)
            if doc.title:
                results.append(doc)
        return results

    async def get(self, source_id: str) -> SourceDocument | None:
        """source_id should be 'owner/repo' format."""
        client = await self._get_client()
        try:
            resp = await client.get(f"/repos/{source_id}")
            resp.raise_for_status()
            return _parse_repo(resp.json())
        except httpx.HTTPStatusError:
            return None

    async def expand(self, source_id: str, limit: int = 10) -> list[SourceDocument]:
        """GitHub doesn't have citation expansion. Return empty."""
        return []

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None
```

### File: src/packs/research/connectors/web_search.py

```python
"""
Web search connector using Tavily.

Used for: finding technical blogs, documentation, vendor analysis
that doesn't appear in scholarly databases.
"""

from __future__ import annotations
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.core.types import SourceDocument
from src.domain_pack import SourceConnector

TAVILY_URL = "https://api.tavily.com/search"


class WebSearchConnector(SourceConnector):
    def __init__(self, api_key: str = ""):
        self.api_key = api_key

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.ConnectError)),
    )
    async def search(self, query: str, limit: int = 10, **kwargs) -> list[SourceDocument]:
        if not self.api_key:
            return []
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(TAVILY_URL, json={
                "api_key": self.api_key,
                "query": query,
                "max_results": min(limit, 20),
                "include_answer": False,
                "search_depth": "advanced",
            })
            resp.raise_for_status()
            data = resp.json()
        
        results = []
        for item in data.get("results", []):
            doc = SourceDocument(
                source_id=item.get("url", ""),
                source_family="web",
                source_url=item.get("url", ""),
                title=item.get("title", ""),
                abstract=item.get("content", "")[:2000],
                publication_date=None,
                source_tier=3,  # web content = Tier 3 by default
                domain="research",
                metadata={"score": item.get("score", 0), "source_type": "web_article"},
            )
            if doc.title:
                results.append(doc)
        return results

    async def get(self, source_id: str) -> SourceDocument | None:
        return None  # Tavily doesn't support single-URL fetch

    async def expand(self, source_id: str, limit: int = 10) -> list[SourceDocument]:
        return []  # No expansion for web search
```

### File: src/packs/research/connectors/__init__.py

```python
from src.packs.research.connectors.openalex import OpenAlexConnector
from src.packs.research.connectors.semantic_scholar import SemanticScholarConnector
from src.packs.research.connectors.arxiv import ArxivConnector
from src.packs.research.connectors.github import GitHubConnector
from src.packs.research.connectors.web_search import WebSearchConnector
```

### Test: tests/packs/research/test_connectors.py

Write tests that use the fixtures/ JSON files for offline testing plus a few optional live API tests (marked with `@pytest.mark.live`):

```python
"""Tests for research source connectors.

Tests marked @pytest.mark.live make real API calls and require API keys.
Run with: pytest -m live tests/packs/research/test_connectors.py
All other tests use fixtures and run offline.
"""
import json
import pytest
import pytest_asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

from src.core.types import SourceDocument
from src.packs.research.connectors.openalex import OpenAlexConnector, _parse_work, _invert_abstract
from src.packs.research.connectors.semantic_scholar import SemanticScholarConnector, _parse_paper
from src.packs.research.connectors.arxiv import ArxivConnector


class TestOpenAlexParsing:
    def test_invert_abstract(self):
        inverted = {"The": [0], "cat": [1], "sat": [2]}
        assert _invert_abstract(inverted) == "The cat sat"
    
    def test_invert_abstract_none(self):
        assert _invert_abstract(None) == ""

    def test_parse_work_returns_source_document(self):
        work = {
            "id": "https://openalex.org/W123",
            "title": "Test Paper",
            "abstract_inverted_index": {"A": [0], "test": [1]},
            "publication_date": "2024-01-15",
            "cited_by_count": 100,
            "type": "journal-article",
            "authorships": [{"author": {"display_name": "Alice"}}],
            "primary_location": {"is_oa": True},
            "concepts": [{"display_name": "Machine Learning"}],
            "doi": "https://doi.org/10.1234/test",
            "is_oa": True,
        }
        doc = _parse_work(work)
        assert isinstance(doc, SourceDocument)
        assert doc.title == "Test Paper"
        assert doc.abstract == "A test"
        assert doc.source_family == "scholarly"
        assert doc.source_tier == 1  # journal article
        assert "Alice" in doc.authors


class TestSemanticScholarParsing:
    def test_parse_paper_returns_source_document(self):
        paper = {
            "paperId": "abc123",
            "title": "Test Paper",
            "abstract": "This is a test abstract",
            "year": 2024,
            "citationCount": 50,
            "influentialCitationCount": 10,
            "authors": [{"name": "Bob"}],
            "externalIds": {"DOI": "10.1234/test"},
            "publicationDate": "2024-03-20",
            "tldr": {"text": "A short summary"},
            "embedding": {"model": "specter2", "vector": [0.1, 0.2, 0.3]},
        }
        doc = _parse_paper(paper)
        assert isinstance(doc, SourceDocument)
        assert doc.title == "Test Paper"
        assert doc.precomputed_embedding == [0.1, 0.2, 0.3]
        assert doc.metadata["tldr"] == "A short summary"
        assert doc.metadata["doi"] == "10.1234/test"

    def test_parse_paper_handles_missing_fields(self):
        paper = {"paperId": "xyz", "title": "Minimal"}
        doc = _parse_paper(paper)
        assert doc.title == "Minimal"
        assert doc.abstract == ""
        assert doc.precomputed_embedding is None


class TestConnectorInterface:
    """Verify all connectors return SourceDocument instances."""
    
    def test_openalex_parse_returns_correct_type(self):
        work = {"id": "https://openalex.org/W1", "title": "Test"}
        doc = _parse_work(work)
        assert isinstance(doc, SourceDocument)
        assert doc.domain == "research"
    
    def test_s2_parse_returns_correct_type(self):
        paper = {"paperId": "abc", "title": "Test"}
        doc = _parse_paper(paper)
        assert isinstance(doc, SourceDocument)
        assert doc.domain == "research"
```

Create fixture files in `fixtures/` with sample API responses for offline testing. Save actual responses from the APIs during development.

**Run tests:** `pytest tests/packs/research/test_connectors.py -v`

**Done when:** All connectors parse API responses into SourceDocuments. Rate limiting and retry logic is in place. Fixture-based tests pass offline.

---

## Phase 4: Corpus manager + embeddings

### File: src/shared/corpus.py

```python
"""
Corpus manager: dedup, expansion, tiering.

Takes raw SourceDocuments from connectors, deduplicates them,
optionally expands via citations, and assigns source tiers.
Domain-pack-aware but not domain-specific.
"""

from __future__ import annotations
import logging
from difflib import SequenceMatcher

from src.core.types import SourceDocument
from src.domain_pack import DomainPack, SourceConnector

logger = logging.getLogger(__name__)


def _normalize_title(title: str) -> str:
    """Normalize title for fuzzy matching."""
    return title.lower().strip().rstrip(".")


def _title_similarity(a: str, b: str) -> float:
    """Compute title similarity ratio."""
    return SequenceMatcher(None, _normalize_title(a), _normalize_title(b)).ratio()


def _get_dedup_key(doc: SourceDocument) -> str | None:
    """Get a dedup key from DOI or other unique identifier."""
    doi = doc.metadata.get("doi", "")
    if doi:
        return f"doi:{doi.lower().strip()}"
    openalex_id = doc.metadata.get("openalex_id", "")
    if openalex_id:
        return f"oa:{openalex_id}"
    s2_id = doc.metadata.get("s2_id", "")
    if s2_id:
        return f"s2:{s2_id}"
    return None


def deduplicate(documents: list[SourceDocument], title_threshold: float = 0.9) -> list[SourceDocument]:
    """Deduplicate documents by DOI, then by fuzzy title match.
    
    When duplicates are found, keep the one with more metadata
    (longer abstract, more authors, lower source_tier number).
    """
    seen_keys: dict[str, SourceDocument] = {}
    no_key: list[SourceDocument] = []
    
    for doc in documents:
        key = _get_dedup_key(doc)
        if key:
            if key in seen_keys:
                existing = seen_keys[key]
                # Keep the richer document
                if len(doc.abstract) > len(existing.abstract) or doc.source_tier < existing.source_tier:
                    seen_keys[key] = doc
            else:
                seen_keys[key] = doc
        else:
            no_key.append(doc)
    
    # Fuzzy dedup on no-key docs by title
    deduped_no_key: list[SourceDocument] = []
    for doc in no_key:
        is_dup = False
        for existing in list(seen_keys.values()) + deduped_no_key:
            if _title_similarity(doc.title, existing.title) > title_threshold:
                is_dup = True
                # Keep richer version
                if len(doc.abstract) > len(existing.abstract):
                    if existing in deduped_no_key:
                        deduped_no_key.remove(existing)
                        deduped_no_key.append(doc)
                break
        if not is_dup:
            deduped_no_key.append(doc)
    
    result = list(seen_keys.values()) + deduped_no_key
    logger.info(f"Dedup: {len(documents)} -> {len(result)} documents")
    return result


async def expand_corpus(
    documents: list[SourceDocument],
    connectors: list[SourceConnector],
    budget: int = 50,
    existing_ids: set[str] | None = None,
) -> list[SourceDocument]:
    """1-hop expansion: find cited-by papers for top documents.
    
    Expands from the most-cited documents first, up to budget limit.
    Skips documents already in corpus.
    """
    if existing_ids is None:
        existing_ids = {doc.source_id for doc in documents}
    
    # Sort by citation count descending to expand from most-cited first
    sorted_docs = sorted(
        documents,
        key=lambda d: d.metadata.get("cited_by_count", 0) or d.metadata.get("citation_count", 0),
        reverse=True,
    )
    
    new_docs: list[SourceDocument] = []
    for doc in sorted_docs[:20]:  # expand from top 20 most-cited
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
                logger.warning(f"Expansion failed for {doc.source_id}: {e}")
                continue
    
    logger.info(f"Expansion: added {len(new_docs)} documents")
    return new_docs


def assign_tiers(documents: list[SourceDocument], pack: DomainPack) -> list[SourceDocument]:
    """Assign source tiers using the domain pack's classification logic."""
    for doc in documents:
        doc.source_tier = pack.classify_tier(doc)
    return documents


def corpus_stats(documents: list[SourceDocument]) -> dict:
    """Compute summary statistics for the corpus."""
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
```

### File: src/shared/embeddings.py

```python
"""
Embedding generation and caching.

Uses sentence-transformers/all-MiniLM-L6-v2 as default model.
Falls back to pre-computed embeddings (e.g., SPECTER2 from Semantic Scholar)
when available.
"""

from __future__ import annotations
import logging
import numpy as np
from typing import Any

logger = logging.getLogger(__name__)

# Lazy-loaded to avoid import overhead when not needed
_model: Any = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Loaded embedding model: all-MiniLM-L6-v2")
    return _model


def embed_texts(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Generate embeddings for a list of texts.
    
    Returns numpy array of shape (len(texts), embedding_dim).
    """
    model = _get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return np.array(embeddings)


def embed_single(text: str) -> list[float]:
    """Generate embedding for a single text. Returns list of floats."""
    model = _get_model()
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity matrix.
    
    Input: (N, D) array. Output: (N, N) similarity matrix.
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # avoid division by zero
    normalized = embeddings / norms
    return np.dot(normalized, normalized.T)
```

### File: src/shared/extraction.py

```python
"""
LLM extraction pipeline.

Extracts structured knowledge (nodes + edges) from SourceDocuments
using the domain pack's extraction prompt. Uses Haiku for batch
extraction (cheap, fast) and returns ExtractionResult objects.
"""

from __future__ import annotations
import json
import logging
import os
from typing import Any

import anthropic

from src.core.types import (
    Node, Edge, EdgeMeta, EdgeType, NodeType, SourceDocument, ExtractionResult,
)
from src.domain_pack import DomainSchema

logger = logging.getLogger(__name__)

EXTRACTION_MODEL = "claude-haiku-4-5-20241022"
MAX_BATCH_SIZE = 5  # abstracts per LLM call


def _build_batch_prompt(schema: DomainSchema, documents: list[SourceDocument]) -> str:
    """Build a batch extraction prompt for multiple documents."""
    doc_blocks = []
    for i, doc in enumerate(documents):
        doc_blocks.append(f"--- DOCUMENT {i+1} ---\nTitle: {doc.title}\nAbstract: {doc.abstract or '(no abstract)'}\n")
    
    return (
        "You are extracting structured knowledge from multiple documents.\n"
        "For EACH document, extract assertions, entities, and relationships.\n\n"
        f"{schema.extraction_prompt.split('Text to extract from:')[0]}"
        "\nProcess ALL documents below. Return a JSON array with one object per document.\n"
        "Each object has the same format: {assertions, entities, relationships}.\n"
        "Respond ONLY with a valid JSON array. No markdown, no preamble.\n\n"
        + "\n".join(doc_blocks)
    )


def _map_entity_type(domain_type: str, schema: DomainSchema) -> NodeType:
    """Map a domain-specific entity type to a universal NodeType."""
    universal = schema.entity_types.get(domain_type, "")
    try:
        return NodeType(universal)
    except ValueError:
        # Fallback mapping
        type_map = {
            "actor": NodeType.ACTOR,
            "concept": NodeType.CONCEPT,
            "artifact": NodeType.ARTIFACT,
            "metric": NodeType.METRIC,
        }
        return type_map.get(domain_type, NodeType.CONCEPT)


def _map_edge_type(domain_type: str, schema: DomainSchema) -> EdgeType:
    """Map a domain-specific relationship type to a universal EdgeType."""
    universal = schema.edge_types.get(domain_type, "")
    try:
        return EdgeType(universal)
    except ValueError:
        # Direct match attempt
        try:
            return EdgeType(domain_type)
        except ValueError:
            return EdgeType.CO_OCCURS


def _parse_extraction(
    raw: dict,
    doc: SourceDocument,
    schema: DomainSchema,
) -> ExtractionResult:
    """Parse LLM extraction output into universal types."""
    nodes: list[Node] = []
    edges: list[Edge] = []
    node_name_to_id: dict[str, str] = {}
    
    # Create a node for the source document itself
    doc_node = Node(
        id=doc.id,
        node_type=NodeType.SOURCE_DOCUMENT,
        name=doc.title,
        description=doc.abstract[:500] if doc.abstract else "",
        domain=doc.domain,
        properties={"source_url": doc.source_url, "publication_date": doc.publication_date},
    )
    nodes.append(doc_node)
    
    base_meta = EdgeMeta(
        source_tier=doc.source_tier,
        source_family=doc.source_family,
        source_url=doc.source_url,
        source_id=doc.source_id,
        extraction_confidence=0.8,  # default; LLM can adjust
        extraction_model=EXTRACTION_MODEL,
        timestamp=doc.publication_date or "",
        provenance="abstract",
        domain=doc.domain,
    )
    
    # Parse assertions
    for assertion in raw.get("assertions", []):
        text = assertion.get("text", "").strip()
        if not text:
            continue
        node = Node(
            node_type=NodeType.ASSERTION,
            name=text[:100],
            description=text,
            domain=doc.domain,
            properties={
                "conditions": assertion.get("conditions", ""),
                "polarity": assertion.get("polarity", "neutral"),
                "full_text": text,
            },
        )
        nodes.append(node)
        node_name_to_id[text[:100]] = node.id
        
        # Link assertion to source document
        edges.append(Edge(
            source_node_id=doc.id,
            target_node_id=node.id,
            edge_type=EdgeType.ASSERTS,
            meta=base_meta,
        ))
    
    # Parse entities
    for entity in raw.get("entities", []):
        name = entity.get("name", "").strip()
        if not name:
            continue
        entity_type = entity.get("entity_type", "concept")
        node = Node(
            node_type=_map_entity_type(entity_type, schema),
            name=name,
            description=entity.get("description", ""),
            domain=doc.domain,
        )
        nodes.append(node)
        node_name_to_id[name] = node.id
    
    # Parse relationships
    for rel in raw.get("relationships", []):
        source_name = rel.get("source_name", "").strip()
        target_name = rel.get("target_name", "").strip()
        rel_type = rel.get("relationship_type", "co_occurs")
        
        source_id = node_name_to_id.get(source_name)
        target_id = node_name_to_id.get(target_name)
        
        if source_id and target_id:
            edges.append(Edge(
                source_node_id=source_id,
                target_node_id=target_id,
                edge_type=_map_edge_type(rel_type, schema),
                meta=base_meta,
            ))
    
    return ExtractionResult(
        source_document_id=doc.id,
        nodes=nodes,
        edges=edges,
    )


async def extract_batch(
    documents: list[SourceDocument],
    schema: DomainSchema,
    api_key: str | None = None,
) -> list[ExtractionResult]:
    """Extract structured knowledge from a batch of documents.
    
    Processes documents in batches of MAX_BATCH_SIZE, calling Haiku
    for each batch. Returns one ExtractionResult per document.
    """
    if not api_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")
    
    client = anthropic.AsyncAnthropic(api_key=api_key)
    results: list[ExtractionResult] = []
    
    # Filter to documents with extractable text
    extractable = [d for d in documents if d.abstract or d.full_text]
    skipped = len(documents) - len(extractable)
    if skipped > 0:
        logger.info(f"Skipping {skipped} documents without text")
    
    # Process in batches
    for i in range(0, len(extractable), MAX_BATCH_SIZE):
        batch = extractable[i:i + MAX_BATCH_SIZE]
        prompt = _build_batch_prompt(schema, batch)
        
        try:
            response = await client.messages.create(
                model=EXTRACTION_MODEL,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            
            raw_text = response.content[0].text.strip()
            # Handle both single object and array responses
            if raw_text.startswith("["):
                parsed = json.loads(raw_text)
            else:
                parsed = [json.loads(raw_text)]
            
            for j, doc in enumerate(batch):
                if j < len(parsed):
                    result = _parse_extraction(parsed[j], doc, schema)
                else:
                    result = ExtractionResult(source_document_id=doc.id)
                results.append(result)
                
        except (json.JSONDecodeError, anthropic.APIError) as e:
            logger.error(f"Extraction failed for batch {i//MAX_BATCH_SIZE}: {e}")
            # Return empty results for failed batch
            for doc in batch:
                results.append(ExtractionResult(source_document_id=doc.id))
    
    total_nodes = sum(len(r.nodes) for r in results)
    total_edges = sum(len(r.edges) for r in results)
    logger.info(f"Extraction complete: {len(results)} docs, {total_nodes} nodes, {total_edges} edges")
    
    return results
```

### Tests for Phase 4-5: tests/shared/

```python
# tests/shared/test_corpus.py
"""Tests for corpus manager."""
import pytest
from src.core.types import SourceDocument
from src.shared.corpus import deduplicate, corpus_stats


class TestDedup:
    def test_dedup_by_doi(self):
        docs = [
            SourceDocument(source_id="a", title="Paper A", metadata={"doi": "10.1234/test"}),
            SourceDocument(source_id="b", title="Paper A copy", metadata={"doi": "10.1234/test"}),
        ]
        result = deduplicate(docs)
        assert len(result) == 1

    def test_dedup_by_title_fuzzy(self):
        docs = [
            SourceDocument(source_id="a", title="Attention Is All You Need"),
            SourceDocument(source_id="b", title="Attention is All You Need."),
        ]
        result = deduplicate(docs)
        assert len(result) == 1

    def test_keeps_different_papers(self):
        docs = [
            SourceDocument(source_id="a", title="Paper Alpha", metadata={"doi": "10.1/alpha"}),
            SourceDocument(source_id="b", title="Paper Beta", metadata={"doi": "10.1/beta"}),
        ]
        result = deduplicate(docs)
        assert len(result) == 2

    def test_keeps_richer_duplicate(self):
        docs = [
            SourceDocument(source_id="a", title="Test", abstract="", metadata={"doi": "10.1/x"}),
            SourceDocument(source_id="b", title="Test", abstract="A full abstract here", metadata={"doi": "10.1/x"}),
        ]
        result = deduplicate(docs)
        assert len(result) == 1
        assert result[0].abstract == "A full abstract here"


class TestCorpusStats:
    def test_counts_tiers(self):
        docs = [
            SourceDocument(source_tier=1, source_family="scholarly"),
            SourceDocument(source_tier=1, source_family="scholarly"),
            SourceDocument(source_tier=2, source_family="code"),
            SourceDocument(source_tier=3, source_family="web"),
        ]
        stats = corpus_stats(docs)
        assert stats["total_documents"] == 4
        assert stats["documents_per_tier"][1] == 2
        assert stats["documents_per_tier"][2] == 1
```

```python
# tests/shared/test_extraction.py
"""Tests for LLM extraction pipeline."""
import pytest
from src.core.types import NodeType, EdgeType, SourceDocument
from src.shared.extraction import _parse_extraction, _map_entity_type, _map_edge_type
from src.packs.research.schema import get_research_schema


class TestParseExtraction:
    def setup_method(self):
        self.schema = get_research_schema()
        self.doc = SourceDocument(
            id="doc_1", source_id="W123", source_family="scholarly",
            source_url="https://example.com", title="Test Paper",
            abstract="A test abstract", source_tier=1, domain="research",
        )

    def test_parses_assertions(self):
        raw = {
            "assertions": [{"text": "Transformers are better than RNNs", "polarity": "positive"}],
            "entities": [],
            "relationships": [],
        }
        result = _parse_extraction(raw, self.doc, self.schema)
        assertion_nodes = [n for n in result.nodes if n.node_type == NodeType.ASSERTION]
        assert len(assertion_nodes) == 1
        assert "Transformers" in assertion_nodes[0].description

    def test_parses_entities(self):
        raw = {
            "assertions": [],
            "entities": [{"name": "BERT", "entity_type": "concept", "description": "A language model"}],
            "relationships": [],
        }
        result = _parse_extraction(raw, self.doc, self.schema)
        concept_nodes = [n for n in result.nodes if n.node_type == NodeType.CONCEPT]
        assert len(concept_nodes) == 1
        assert concept_nodes[0].name == "BERT"

    def test_creates_source_document_node(self):
        raw = {"assertions": [], "entities": [], "relationships": []}
        result = _parse_extraction(raw, self.doc, self.schema)
        doc_nodes = [n for n in result.nodes if n.node_type == NodeType.SOURCE_DOCUMENT]
        assert len(doc_nodes) == 1

    def test_links_assertions_to_document(self):
        raw = {
            "assertions": [{"text": "Test claim", "polarity": "neutral"}],
            "entities": [],
            "relationships": [],
        }
        result = _parse_extraction(raw, self.doc, self.schema)
        asserts_edges = [e for e in result.edges if e.edge_type == EdgeType.ASSERTS]
        assert len(asserts_edges) == 1

    def test_handles_empty_extraction(self):
        raw = {"assertions": [], "entities": [], "relationships": []}
        result = _parse_extraction(raw, self.doc, self.schema)
        assert result.source_document_id == "doc_1"
        # Should still have the document node
        assert len(result.nodes) == 1

    def test_edges_carry_full_metadata(self):
        raw = {
            "assertions": [{"text": "Test claim"}],
            "entities": [],
            "relationships": [],
        }
        result = _parse_extraction(raw, self.doc, self.schema)
        for edge in result.edges:
            assert edge.meta.source_tier == 1
            assert edge.meta.source_family == "scholarly"
            assert edge.meta.domain == "research"
```

**Run all Part 1 tests:** `pytest tests/core/ tests/packs/ tests/shared/ -v`

**Done when all pass.** Part 1 is the foundation — types, domain pack, connectors, corpus, extraction. All domain-agnostic abstractions are in place. Every module returns universal types.

---

## What Part 2 covers (separate document)

Part 2 implements the core engine:

- Phase 6: Graph construction (src/core/graph.py) — NetworkX builder, entity resolution, serialization
- Phase 7: Bridge detection + contradiction detection
- Phase 8: Drift detection + gap detection
- Phase 9: Verifier + promotion gate
- Phase 10: Report generator + agent.py integration + D3.js graph visualization

Part 2 depends on Part 1 being complete and tested.
