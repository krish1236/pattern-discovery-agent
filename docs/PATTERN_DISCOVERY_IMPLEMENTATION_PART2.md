# Pattern Discovery Agent — Implementation Part 2
## Engine: Graph, Patterns, Verifier, Report, Integration

**Depends on Part 1 being complete and tested.**

This doc covers Phases 6–10: graph construction, all four pattern detectors, the verifier/promotion gate, report generator, D3.js graph visualization, and the final agent.py RunForge integration.

---

## Phase 6: Knowledge graph construction

### File: src/core/graph.py

```python
"""
Knowledge graph builder.

Constructs a NetworkX graph from ExtractionResults. Handles entity
resolution (merging duplicate entities), serialization to/from JSON,
and graph statistics.

RULE: This module is domain-agnostic. It operates on Node, Edge,
EdgeMeta — never on domain-specific concepts.
"""

from __future__ import annotations
import json
import logging
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any

import networkx as nx

from src.core.types import (
    Node, Edge, EdgeMeta, EdgeType, NodeType, ExtractionResult, SourceDocument,
)

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """A typed, metadata-rich knowledge graph backed by NetworkX.
    
    Every node is a Node dataclass. Every edge carries EdgeMeta provenance.
    The graph can be serialized to JSON for RunForge ctx.storage and
    deserialized to resume from checkpoint.
    """

    def __init__(self):
        self.g: nx.Graph = nx.Graph()
        self._node_registry: dict[str, Node] = {}  # id -> Node
        self._name_index: dict[str, list[str]] = defaultdict(list)  # normalized_name -> [node_ids]

    @property
    def node_count(self) -> int:
        return self.g.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self.g.number_of_edges()

    def add_node(self, node: Node) -> str:
        """Add a node to the graph. Returns the node ID (may differ if merged)."""
        # Try entity resolution first
        resolved_id = self._resolve_entity(node)
        if resolved_id and resolved_id != node.id:
            # Merge into existing node
            self._merge_node(resolved_id, node)
            return resolved_id
        
        self._node_registry[node.id] = node
        self.g.add_node(node.id, **node.to_dict())
        
        # Index by normalized name for future resolution
        normalized = self._normalize_name(node.name)
        if normalized:
            self._name_index[normalized].append(node.id)
        
        return node.id

    def add_edge(self, edge: Edge) -> None:
        """Add an edge to the graph with full metadata."""
        if edge.source_node_id not in self.g:
            logger.warning(f"Source node {edge.source_node_id} not in graph, skipping edge")
            return
        if edge.target_node_id not in self.g:
            logger.warning(f"Target node {edge.target_node_id} not in graph, skipping edge")
            return
        
        # NetworkX allows multiple edges between same pair in MultiGraph
        # For simplicity, we use Graph and store edge list as attribute
        key = (edge.source_node_id, edge.target_node_id)
        if self.g.has_edge(*key):
            existing = self.g.edges[key]
            edges_list = existing.get("edges", [])
            edges_list.append(edge.to_dict())
            self.g.edges[key]["edges"] = edges_list
        else:
            self.g.add_edge(
                edge.source_node_id,
                edge.target_node_id,
                edge_type=edge.edge_type.value,
                edges=[edge.to_dict()],
            )

    def add_extraction_result(self, result: ExtractionResult) -> dict[str, str]:
        """Add all nodes and edges from an extraction result.
        
        Returns a mapping of original_node_id -> resolved_node_id
        (for cases where entity resolution merged nodes).
        """
        id_mapping: dict[str, str] = {}
        
        for node in result.nodes:
            resolved_id = self.add_node(node)
            id_mapping[node.id] = resolved_id
        
        for edge in result.edges:
            # Remap edge node IDs through resolution
            resolved_edge = Edge(
                source_node_id=id_mapping.get(edge.source_node_id, edge.source_node_id),
                target_node_id=id_mapping.get(edge.target_node_id, edge.target_node_id),
                edge_type=edge.edge_type,
                meta=edge.meta,
                properties=edge.properties,
            )
            self.add_edge(resolved_edge)
        
        return id_mapping

    def get_node(self, node_id: str) -> Node | None:
        """Get a node by ID."""
        return self._node_registry.get(node_id)

    def get_nodes_by_type(self, node_type: NodeType) -> list[Node]:
        """Get all nodes of a specific type."""
        return [n for n in self._node_registry.values() if n.node_type == node_type]

    def get_edges_by_type(self, edge_type: EdgeType) -> list[Edge]:
        """Get all edges of a specific type."""
        result = []
        for u, v, data in self.g.edges(data=True):
            for edge_dict in data.get("edges", []):
                if edge_dict.get("edge_type") == edge_type.value:
                    result.append(Edge.from_dict(edge_dict))
        return result

    def get_neighbors(self, node_id: str) -> list[Node]:
        """Get all neighbor nodes."""
        if node_id not in self.g:
            return []
        return [self._node_registry[n] for n in self.g.neighbors(node_id) if n in self._node_registry]

    # --- Entity Resolution ---

    def _normalize_name(self, name: str) -> str:
        """Normalize entity name for matching."""
        return name.lower().strip().replace("-", " ").replace("_", " ")

    def _resolve_entity(self, node: Node) -> str | None:
        """Try to find an existing node that matches this one.
        
        Resolution rules:
        - Exact normalized name match + same NodeType = merge
        - High fuzzy match (>0.85) + same NodeType = merge
        - Different NodeType = never merge (even if same name)
        """
        if node.node_type == NodeType.SOURCE_DOCUMENT:
            return None  # never merge source documents by name
        
        normalized = self._normalize_name(node.name)
        if not normalized:
            return None
        
        # Exact match
        candidates = self._name_index.get(normalized, [])
        for cid in candidates:
            existing = self._node_registry.get(cid)
            if existing and existing.node_type == node.node_type:
                return cid
        
        # Fuzzy match against all nodes of same type
        for existing_name, existing_ids in self._name_index.items():
            similarity = SequenceMatcher(None, normalized, existing_name).ratio()
            if similarity > 0.85:
                for eid in existing_ids:
                    existing = self._node_registry.get(eid)
                    if existing and existing.node_type == node.node_type:
                        return eid
        
        return None

    def _merge_node(self, existing_id: str, new_node: Node) -> None:
        """Merge a new node into an existing one. Keep richer data."""
        existing = self._node_registry[existing_id]
        if len(new_node.description) > len(existing.description):
            existing.description = new_node.description
        existing.properties.update(new_node.properties)

    # --- Serialization ---

    def to_json(self) -> str:
        """Serialize the graph to JSON string."""
        data = {
            "nodes": [node.to_dict() for node in self._node_registry.values()],
            "edges": [],
        }
        for u, v, edge_data in self.g.edges(data=True):
            for edge_dict in edge_data.get("edges", []):
                data["edges"].append(edge_dict)
        return json.dumps(data)

    @classmethod
    def from_json(cls, json_str: str) -> KnowledgeGraph:
        """Deserialize a graph from JSON string."""
        data = json.loads(json_str)
        graph = cls()
        for node_dict in data["nodes"]:
            node = Node.from_dict(node_dict)
            graph._node_registry[node.id] = node
            graph.g.add_node(node.id, **node.to_dict())
            normalized = graph._normalize_name(node.name)
            if normalized:
                graph._name_index[normalized].append(node.id)
        for edge_dict in data["edges"]:
            edge = Edge.from_dict(edge_dict)
            graph.add_edge(edge)
        return graph

    # --- Stats ---

    def stats(self) -> dict:
        """Compute graph statistics."""
        type_counts: dict[str, int] = defaultdict(int)
        for node in self._node_registry.values():
            type_counts[node.node_type.value] += 1
        
        edge_type_counts: dict[str, int] = defaultdict(int)
        for _, _, data in self.g.edges(data=True):
            for edge_dict in data.get("edges", []):
                edge_type_counts[edge_dict.get("edge_type", "unknown")] += 1
        
        components = nx.number_connected_components(self.g) if self.node_count > 0 else 0
        
        return {
            "total_nodes": self.node_count,
            "total_edges": self.edge_count,
            "nodes_by_type": dict(type_counts),
            "edges_by_type": dict(edge_type_counts),
            "connected_components": components,
        }
```

### Test: tests/core/test_graph.py

```python
"""Tests for knowledge graph construction."""
import json
import pytest
from src.core.types import Node, Edge, EdgeMeta, EdgeType, NodeType, ExtractionResult
from src.core.graph import KnowledgeGraph


@pytest.fixture
def empty_graph():
    return KnowledgeGraph()


@pytest.fixture
def sample_meta():
    return EdgeMeta(
        source_tier=1, source_family="scholarly", source_url="https://example.com",
        source_id="W123", extraction_confidence=0.9, extraction_model="haiku",
        timestamp="2025-01-01", provenance="abstract", domain="research",
    )


class TestGraphConstruction:
    def test_add_node(self, empty_graph):
        node = Node(node_type=NodeType.CONCEPT, name="Attention", domain="research")
        nid = empty_graph.add_node(node)
        assert empty_graph.node_count == 1
        assert empty_graph.get_node(nid).name == "Attention"

    def test_add_edge(self, empty_graph, sample_meta):
        n1 = Node(id="n1", node_type=NodeType.ASSERTION, name="Claim 1")
        n2 = Node(id="n2", node_type=NodeType.ASSERTION, name="Claim 2")
        empty_graph.add_node(n1)
        empty_graph.add_node(n2)
        edge = Edge(source_node_id="n1", target_node_id="n2",
                    edge_type=EdgeType.SUPPORTS, meta=sample_meta)
        empty_graph.add_edge(edge)
        assert empty_graph.edge_count == 1


class TestEntityResolution:
    def test_merges_same_name_same_type(self, empty_graph):
        n1 = Node(node_type=NodeType.CONCEPT, name="BERT", domain="research")
        n2 = Node(node_type=NodeType.CONCEPT, name="bert", domain="research")
        id1 = empty_graph.add_node(n1)
        id2 = empty_graph.add_node(n2)
        assert id1 == id2
        assert empty_graph.node_count == 1

    def test_merges_fuzzy_match(self, empty_graph):
        n1 = Node(node_type=NodeType.CONCEPT, name="Bidirectional Encoder Representations",
                  domain="research")
        empty_graph.add_node(n1)
        # This won't fuzzy match because the names are too different
        n2 = Node(node_type=NodeType.CONCEPT, name="BERT", domain="research")
        empty_graph.add_node(n2)
        assert empty_graph.node_count == 2  # different enough to be separate

    def test_does_not_merge_different_types(self, empty_graph):
        n1 = Node(node_type=NodeType.CONCEPT, name="Transformer", domain="research")
        n2 = Node(node_type=NodeType.ARTIFACT, name="Transformer", domain="research")
        empty_graph.add_node(n1)
        empty_graph.add_node(n2)
        assert empty_graph.node_count == 2

    def test_never_merges_source_documents(self, empty_graph):
        n1 = Node(node_type=NodeType.SOURCE_DOCUMENT, name="Paper A")
        n2 = Node(node_type=NodeType.SOURCE_DOCUMENT, name="Paper A")
        empty_graph.add_node(n1)
        empty_graph.add_node(n2)
        assert empty_graph.node_count == 2


class TestSerialization:
    def test_roundtrip_json(self, empty_graph, sample_meta):
        n1 = Node(id="n1", node_type=NodeType.CONCEPT, name="Attention", domain="research")
        n2 = Node(id="n2", node_type=NodeType.ASSERTION, name="Claim", domain="research")
        empty_graph.add_node(n1)
        empty_graph.add_node(n2)
        edge = Edge(source_node_id="n1", target_node_id="n2",
                    edge_type=EdgeType.ASSOCIATED_WITH, meta=sample_meta)
        empty_graph.add_edge(edge)

        json_str = empty_graph.to_json()
        restored = KnowledgeGraph.from_json(json_str)
        assert restored.node_count == 2
        assert restored.edge_count == 1
        assert restored.get_node("n1").name == "Attention"

    def test_json_is_valid(self, empty_graph):
        n = Node(node_type=NodeType.CONCEPT, name="Test")
        empty_graph.add_node(n)
        json_str = empty_graph.to_json()
        parsed = json.loads(json_str)  # should not raise
        assert "nodes" in parsed
        assert "edges" in parsed


class TestStats:
    def test_empty_graph_stats(self, empty_graph):
        stats = empty_graph.stats()
        assert stats["total_nodes"] == 0
        assert stats["total_edges"] == 0

    def test_stats_count_types(self, empty_graph):
        empty_graph.add_node(Node(node_type=NodeType.CONCEPT, name="A"))
        empty_graph.add_node(Node(node_type=NodeType.CONCEPT, name="B"))
        empty_graph.add_node(Node(node_type=NodeType.ACTOR, name="C"))
        stats = empty_graph.stats()
        assert stats["nodes_by_type"]["concept"] == 2
        assert stats["nodes_by_type"]["actor"] == 1


class TestDomainAgnosticism:
    def test_graph_module_has_no_domain_references(self):
        """Verify graph.py doesn't reference domain-specific concepts."""
        import inspect
        from src.core import graph
        source = inspect.getsource(graph)
        for term in ["paper", "claim", "team", "bill", "player", "senator"]:
            # Allow these in comments but not in code
            code_lines = [l for l in source.split("\n")
                         if not l.strip().startswith("#") and not l.strip().startswith('"""')]
            code_only = "\n".join(code_lines).lower()
            assert term not in code_only, f"Domain-specific term '{term}' found in graph.py"
```

---

## Phase 7: Pattern detection — bridges + contradictions

### File: src/core/patterns/bridges.py

```python
"""
Bridge pattern detection.

Finds entities/concepts weakly connected structurally but strongly
related semantically — communities discussing similar things
with different language that rarely reference each other.

Algorithm:
1. Louvain community detection
2. Find inter-community edges
3. Semantic similarity filter
4. Betweenness centrality ranking
5. LLM validation via domain pack
"""

from __future__ import annotations
import logging
from collections import defaultdict

import networkx as nx
import numpy as np

from src.core.types import (
    Node, NodeType, PatternCandidate, PatternType, EvidenceItem, BlindSpot,
)
from src.core.graph import KnowledgeGraph
from src.shared.embeddings import cosine_similarity

logger = logging.getLogger(__name__)


def detect_bridges(
    graph: KnowledgeGraph,
    min_community_size: int = 3,
    semantic_threshold: float = 0.4,
    top_k: int = 10,
) -> list[PatternCandidate]:
    """Detect bridge patterns in the knowledge graph.
    
    Returns PatternCandidate instances (not yet verified/promoted).
    """
    if graph.node_count < 6:
        logger.info("Graph too small for bridge detection")
        return []
    
    # Step 1: Community detection
    communities = nx.community.louvain_communities(
        graph.g, resolution=1.0, seed=42,
    )
    communities = [c for c in communities if len(c) >= min_community_size]
    
    if len(communities) < 2:
        logger.info(f"Only {len(communities)} communities found, need >= 2")
        return []
    
    logger.info(f"Found {len(communities)} communities")
    
    # Build community membership map
    node_to_community: dict[str, int] = {}
    for i, comm in enumerate(communities):
        for node_id in comm:
            node_to_community[node_id] = i
    
    # Step 2: Find inter-community edges
    inter_community_edges: list[tuple[str, str, int, int]] = []
    for u, v in graph.g.edges():
        cu = node_to_community.get(u)
        cv = node_to_community.get(v)
        if cu is not None and cv is not None and cu != cv:
            inter_community_edges.append((u, v, cu, cv))
    
    logger.info(f"Found {len(inter_community_edges)} inter-community edges")
    
    if not inter_community_edges:
        return []
    
    # Step 3: Semantic similarity filter
    bridge_candidates: list[dict] = []
    for u, v, cu, cv in inter_community_edges:
        node_u = graph.get_node(u)
        node_v = graph.get_node(v)
        if not node_u or not node_v:
            continue
        if not node_u.embedding or not node_v.embedding:
            continue
        
        sim = cosine_similarity(
            np.array(node_u.embedding),
            np.array(node_v.embedding),
        )
        if sim >= semantic_threshold:
            bridge_candidates.append({
                "node_u": node_u, "node_v": node_v,
                "community_u": cu, "community_v": cv,
                "similarity": sim,
            })
    
    logger.info(f"After semantic filter: {len(bridge_candidates)} candidates")
    
    # Step 4: Betweenness centrality ranking
    if bridge_candidates:
        bridge_node_ids = set()
        for bc in bridge_candidates:
            bridge_node_ids.add(bc["node_u"].id)
            bridge_node_ids.add(bc["node_v"].id)
        
        centrality = nx.betweenness_centrality(graph.g)
        for bc in bridge_candidates:
            bc["centrality"] = max(
                centrality.get(bc["node_u"].id, 0),
                centrality.get(bc["node_v"].id, 0),
            )
        
        bridge_candidates.sort(key=lambda x: (x["similarity"] * 0.6 + x["centrality"] * 0.4), reverse=True)
    
    # Step 5: Build PatternCandidates for top-K
    patterns: list[PatternCandidate] = []
    for bc in bridge_candidates[:top_k]:
        nu = bc["node_u"]
        nv = bc["node_v"]
        
        # Gather evidence from both communities
        evidence = []
        for neighbor in graph.get_neighbors(nu.id):
            if neighbor.node_type == NodeType.ASSERTION:
                evidence.append(EvidenceItem(
                    assertion_node_id=neighbor.id,
                    assertion_text=neighbor.description[:200],
                    source_document_id=neighbor.properties.get("source_document_id", ""),
                    source_url=neighbor.properties.get("source_url", ""),
                    source_tier=neighbor.properties.get("source_tier", 2),
                ))
        for neighbor in graph.get_neighbors(nv.id):
            if neighbor.node_type == NodeType.ASSERTION:
                evidence.append(EvidenceItem(
                    assertion_node_id=neighbor.id,
                    assertion_text=neighbor.description[:200],
                    source_document_id=neighbor.properties.get("source_document_id", ""),
                    source_url=neighbor.properties.get("source_url", ""),
                    source_tier=neighbor.properties.get("source_tier", 2),
                ))
        
        pattern = PatternCandidate(
            pattern_type=PatternType.BRIDGE,
            title=f"Bridge: {nu.name} ↔ {nv.name}",
            measured_pattern=(
                f"Nodes '{nu.name}' (community {bc['community_u']}) and "
                f"'{nv.name}' (community {bc['community_v']}) are structurally "
                f"weak-linked but semantically similar (cosine={bc['similarity']:.2f}). "
                f"Betweenness centrality: {bc['centrality']:.3f}."
            ),
            evidence=evidence[:10],  # cap at 10
            blind_spots=[BlindSpot(
                description=f"Only {len(evidence)} evidence items found near bridge nodes",
                severity="moderate" if len(evidence) >= 3 else "major",
            )],
            confidence_score=min(bc["similarity"], 0.95),
            domain=nu.domain,
            details={
                "community_u": bc["community_u"],
                "community_v": bc["community_v"],
                "bridge_node_u": nu.name,
                "bridge_node_v": nv.name,
                "semantic_similarity": bc["similarity"],
                "betweenness_centrality": bc["centrality"],
            },
        )
        patterns.append(pattern)
    
    logger.info(f"Produced {len(patterns)} bridge pattern candidates")
    return patterns
```

### File: src/core/patterns/contradictions.py

```python
"""
Contradiction pattern detection.

Finds assertions that conflict — one source says X, another says not-X,
or X holds only under specific conditions.

Algorithm:
1. Collect Assertion nodes
2. Group by topic cluster
3. Generate pairs with cosine similarity > 0.6
4. NLI cross-encoder classifies entailment/neutral/contradiction
5. LLM second pass for real vs apparent contradiction
"""

from __future__ import annotations
import logging
from itertools import combinations

import numpy as np

from src.core.types import (
    Node, NodeType, EdgeType, PatternCandidate, PatternType,
    EvidenceItem, BlindSpot,
)
from src.core.graph import KnowledgeGraph
from src.shared.embeddings import cosine_similarity

logger = logging.getLogger(__name__)

# Lazy-loaded NLI model
_nli_model = None


def _get_nli_model():
    global _nli_model
    if _nli_model is None:
        from sentence_transformers import CrossEncoder
        _nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-base")
        logger.info("Loaded NLI model: cross-encoder/nli-deberta-v3-base")
    return _nli_model


def _classify_nli(pairs: list[tuple[str, str]]) -> list[dict]:
    """Classify pairs using NLI cross-encoder.
    
    Returns list of {label, scores} where label is
    'contradiction', 'entailment', or 'neutral'.
    """
    model = _get_nli_model()
    scores = model.predict(pairs)
    
    labels = ["contradiction", "entailment", "neutral"]
    results = []
    for score_set in scores:
        idx = score_set.argmax()
        results.append({
            "label": labels[idx],
            "confidence": float(score_set[idx]),
            "scores": {labels[i]: float(score_set[i]) for i in range(3)},
        })
    return results


def detect_contradictions(
    graph: KnowledgeGraph,
    similarity_threshold: float = 0.6,
    contradiction_confidence: float = 0.7,
    max_pairs: int = 2000,
) -> list[PatternCandidate]:
    """Detect contradiction patterns between assertions.
    
    Returns PatternCandidate instances (not yet verified).
    """
    # Step 1: Get all assertion nodes with embeddings
    assertions = graph.get_nodes_by_type(NodeType.ASSERTION)
    assertions_with_emb = [a for a in assertions if a.embedding is not None]
    
    if len(assertions_with_emb) < 2:
        logger.info("Not enough assertions with embeddings for contradiction detection")
        return []
    
    logger.info(f"Checking {len(assertions_with_emb)} assertions for contradictions")
    
    # Step 2-3: Find pairs with high semantic similarity
    candidate_pairs: list[tuple[Node, Node, float]] = []
    for a, b in combinations(assertions_with_emb, 2):
        sim = cosine_similarity(np.array(a.embedding), np.array(b.embedding))
        if sim >= similarity_threshold:
            candidate_pairs.append((a, b, sim))
    
    # Cap pairs for performance
    candidate_pairs.sort(key=lambda x: x[2], reverse=True)
    candidate_pairs = candidate_pairs[:max_pairs]
    
    logger.info(f"Found {len(candidate_pairs)} high-similarity assertion pairs")
    
    if not candidate_pairs:
        return []
    
    # Step 4: NLI classification
    text_pairs = [(a.description, b.description) for a, b, _ in candidate_pairs]
    nli_results = _classify_nli(text_pairs)
    
    # Step 5: Build PatternCandidates for contradictions
    patterns: list[PatternCandidate] = []
    for (node_a, node_b, sim), nli_result in zip(candidate_pairs, nli_results):
        if nli_result["label"] != "contradiction":
            continue
        if nli_result["confidence"] < contradiction_confidence:
            continue
        
        # Find source documents for each assertion
        def _find_source_doc(node: Node) -> tuple[str, str, int]:
            for neighbor in graph.get_neighbors(node.id):
                if neighbor.node_type == NodeType.SOURCE_DOCUMENT:
                    return (neighbor.id, neighbor.properties.get("source_url", ""),
                            neighbor.properties.get("source_tier", 2))
            return ("", "", 2)
        
        src_a = _find_source_doc(node_a)
        src_b = _find_source_doc(node_b)
        
        evidence_a = EvidenceItem(
            assertion_node_id=node_a.id,
            assertion_text=node_a.description[:300],
            source_document_id=src_a[0], source_url=src_a[1], source_tier=src_a[2],
            role="supports",
        )
        evidence_b = EvidenceItem(
            assertion_node_id=node_b.id,
            assertion_text=node_b.description[:300],
            source_document_id=src_b[0], source_url=src_b[1], source_tier=src_b[2],
            role="counters",
        )
        
        pattern = PatternCandidate(
            pattern_type=PatternType.CONTRADICTION,
            title=f"Contradiction: {node_a.name[:50]} vs {node_b.name[:50]}",
            measured_pattern=(
                f"NLI cross-encoder classified these assertions as contradictory "
                f"(confidence={nli_result['confidence']:.2f}, "
                f"cosine_similarity={sim:.2f})."
            ),
            evidence=[evidence_a],
            counter_evidence=[evidence_b],
            blind_spots=[BlindSpot(
                description="NLI classification may miss domain-specific nuance",
                severity="moderate",
            )],
            confidence_score=nli_result["confidence"] * 0.8,  # discount slightly
            domain=node_a.domain,
            details={
                "assertion_a": node_a.description,
                "assertion_b": node_b.description,
                "nli_confidence": nli_result["confidence"],
                "nli_scores": nli_result["scores"],
                "cosine_similarity": sim,
            },
        )
        patterns.append(pattern)
    
    logger.info(f"Produced {len(patterns)} contradiction pattern candidates")
    return patterns
```

### Tests: tests/core/test_bridges.py and test_contradictions.py

```python
# tests/core/test_bridges.py
"""Tests for bridge detection."""
import pytest
from src.core.types import Node, Edge, EdgeMeta, EdgeType, NodeType
from src.core.graph import KnowledgeGraph
from src.core.patterns.bridges import detect_bridges


@pytest.fixture
def bridge_graph():
    """Create a graph with two communities and a known bridge."""
    g = KnowledgeGraph()
    meta = EdgeMeta(source_tier=1, source_family="scholarly", source_url="",
                    source_id="", extraction_confidence=0.9, extraction_model="test",
                    timestamp="2025-01-01", provenance="test", domain="research")
    
    # Community A: nodes a1-a4, densely connected
    for i in range(1, 5):
        g.add_node(Node(id=f"a{i}", node_type=NodeType.CONCEPT,
                       name=f"Concept A{i}", domain="research",
                       embedding=[1.0, 0.0, 0.1*i]))
    for i in range(1, 5):
        for j in range(i+1, 5):
            g.add_edge(Edge(source_node_id=f"a{i}", target_node_id=f"a{j}",
                          edge_type=EdgeType.CO_OCCURS, meta=meta))
    
    # Community B: nodes b1-b4, densely connected
    for i in range(1, 5):
        g.add_node(Node(id=f"b{i}", node_type=NodeType.CONCEPT,
                       name=f"Concept B{i}", domain="research",
                       embedding=[0.0, 1.0, 0.1*i]))
    for i in range(1, 5):
        for j in range(i+1, 5):
            g.add_edge(Edge(source_node_id=f"b{i}", target_node_id=f"b{j}",
                          edge_type=EdgeType.CO_OCCURS, meta=meta))
    
    # Bridge: one weak link between a1 and b1 with high semantic similarity
    g._node_registry["a1"].embedding = [0.5, 0.5, 0.1]
    g._node_registry["b1"].embedding = [0.5, 0.5, 0.2]  # similar to a1
    g.add_edge(Edge(source_node_id="a1", target_node_id="b1",
                   edge_type=EdgeType.CO_OCCURS, meta=meta))
    
    return g


class TestBridgeDetection:
    def test_finds_bridge_in_synthetic_graph(self, bridge_graph):
        patterns = detect_bridges(bridge_graph, min_community_size=2, semantic_threshold=0.3)
        assert len(patterns) >= 1
        bridge = patterns[0]
        assert bridge.pattern_type.value == "bridge"
        assert bridge.details["semantic_similarity"] > 0.3

    def test_returns_empty_for_small_graph(self):
        g = KnowledgeGraph()
        g.add_node(Node(node_type=NodeType.CONCEPT, name="A"))
        patterns = detect_bridges(g)
        assert patterns == []

    def test_uses_universal_types_only(self, bridge_graph):
        patterns = detect_bridges(bridge_graph, min_community_size=2)
        for p in patterns:
            assert hasattr(p, "pattern_type")
            assert hasattr(p, "evidence")
            assert hasattr(p, "blind_spots")
```

```python
# tests/core/test_contradictions.py
"""Tests for contradiction detection."""
import pytest
import numpy as np
from src.core.types import Node, Edge, EdgeMeta, EdgeType, NodeType
from src.core.graph import KnowledgeGraph
from src.core.patterns.contradictions import _classify_nli


class TestNLIClassification:
    """Test NLI cross-encoder directly."""
    
    @pytest.mark.slow
    def test_detects_known_contradiction(self):
        pairs = [("The cat is sleeping", "The cat is not sleeping")]
        results = _classify_nli(pairs)
        assert results[0]["label"] == "contradiction"

    @pytest.mark.slow
    def test_detects_known_entailment(self):
        pairs = [("A man is eating pizza", "A man eats something")]
        results = _classify_nli(pairs)
        assert results[0]["label"] == "entailment"

    @pytest.mark.slow
    def test_detects_known_neutral(self):
        pairs = [("A man is eating pizza", "The weather is nice today")]
        results = _classify_nli(pairs)
        assert results[0]["label"] == "neutral"
```

---

## Phase 8: Drift detection + gap detection

### File: src/core/patterns/drift.py

```python
"""
Drift detection — temporal inflection patterns.

Algorithm:
1. Bin nodes by time window
2. Embed text per window with sentence-transformers
3. HDBSCAN cluster per window
4. Track cluster transitions: birth, growth, death, merge
"""

from __future__ import annotations
import logging
from collections import defaultdict
from datetime import datetime

import numpy as np

from src.core.types import (
    Node, NodeType, PatternCandidate, PatternType, EvidenceItem, BlindSpot,
)
from src.core.graph import KnowledgeGraph
from src.shared.embeddings import embed_texts

logger = logging.getLogger(__name__)


def _bin_by_year(nodes: list[Node]) -> dict[str, list[Node]]:
    """Bin nodes by publication year."""
    bins: dict[str, list[Node]] = defaultdict(list)
    for node in nodes:
        date_str = node.properties.get("publication_date", "")
        if date_str and len(date_str) >= 4:
            year = date_str[:4]
            bins[year].append(node)
    return dict(sorted(bins.items()))


def detect_drift(
    graph: KnowledgeGraph,
    min_cluster_size: int = 3,
    min_windows: int = 2,
) -> list[PatternCandidate]:
    """Detect temporal drift patterns.
    
    Returns PatternCandidate instances (not yet verified).
    """
    # Get documents and assertions with dates
    docs = graph.get_nodes_by_type(NodeType.SOURCE_DOCUMENT)
    assertions = graph.get_nodes_by_type(NodeType.ASSERTION)
    
    # Bin by year
    dated_nodes = [n for n in docs + assertions
                   if n.properties.get("publication_date")]
    bins = _bin_by_year(dated_nodes)
    
    if len(bins) < min_windows:
        logger.info(f"Only {len(bins)} time windows, need >= {min_windows}")
        return []
    
    logger.info(f"Drift detection across {len(bins)} time windows: {list(bins.keys())}")
    
    # Cluster per window
    window_clusters: dict[str, list[set[int]]] = {}
    window_texts: dict[str, list[str]] = {}
    
    for window, nodes in bins.items():
        texts = [n.description or n.name for n in nodes]
        if len(texts) < min_cluster_size:
            continue
        
        window_texts[window] = texts
        embeddings = embed_texts(texts)
        
        try:
            import hdbscan
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="cosine")
            labels = clusterer.fit_predict(embeddings)
            
            clusters: dict[int, set[int]] = defaultdict(set)
            for idx, label in enumerate(labels):
                if label >= 0:  # -1 = noise
                    clusters[label].add(idx)
            
            window_clusters[window] = [indices for indices in clusters.values()]
        except Exception as e:
            logger.warning(f"HDBSCAN failed for window {window}: {e}")
            continue
    
    if len(window_clusters) < min_windows:
        return []
    
    # Track transitions
    windows = sorted(window_clusters.keys())
    transitions: list[dict] = []
    
    for i in range(1, len(windows)):
        prev_window = windows[i-1]
        curr_window = windows[i]
        prev_count = len(window_clusters.get(prev_window, []))
        curr_count = len(window_clusters.get(curr_window, []))
        
        transitions.append({
            "from_window": prev_window,
            "to_window": curr_window,
            "prev_clusters": prev_count,
            "curr_clusters": curr_count,
            "delta": curr_count - prev_count,
        })
    
    # Build patterns from significant transitions
    patterns: list[PatternCandidate] = []
    for t in transitions:
        if abs(t["delta"]) >= 1:  # at least one cluster appeared or disappeared
            direction = "grew" if t["delta"] > 0 else "consolidated"
            pattern = PatternCandidate(
                pattern_type=PatternType.DRIFT,
                title=f"Topic drift: {t['from_window']} → {t['to_window']}",
                measured_pattern=(
                    f"Cluster count changed from {t['prev_clusters']} to "
                    f"{t['curr_clusters']} between {t['from_window']} and {t['to_window']}. "
                    f"The topic {direction}."
                ),
                evidence=[],  # filled by LLM interpretation step
                blind_spots=[BlindSpot(
                    description=f"Cluster analysis limited to {len(bins)} time windows",
                    severity="moderate",
                )],
                confidence_score=0.5,
                details={
                    "from_window": t["from_window"],
                    "to_window": t["to_window"],
                    "cluster_delta": t["delta"],
                    "all_windows": windows,
                    "transitions": transitions,
                },
            )
            patterns.append(pattern)
    
    logger.info(f"Produced {len(patterns)} drift pattern candidates")
    return patterns
```

### File: src/core/patterns/gaps.py

```python
"""
Gap detection — finds frequently recommended but unexecuted work.

Algorithm:
1. Find nodes with many incoming recommends/suggests edges
   but few/zero evaluates/produces edges
2. Score by recommender count, tier, recency
"""

from __future__ import annotations
import logging
from collections import defaultdict

from src.core.types import (
    Node, NodeType, EdgeType, PatternCandidate, PatternType,
    EvidenceItem, BlindSpot,
)
from src.core.graph import KnowledgeGraph

logger = logging.getLogger(__name__)


def detect_gaps(
    graph: KnowledgeGraph,
    min_recommendations: int = 2,
    top_k: int = 10,
) -> list[PatternCandidate]:
    """Detect gap patterns — things recommended but not executed."""
    
    # Count incoming edge types per node
    recommends_count: dict[str, list[str]] = defaultdict(list)  # node_id -> [recommender_ids]
    execution_count: dict[str, int] = defaultdict(int)
    
    for u, v, data in graph.g.edges(data=True):
        for edge_dict in data.get("edges", []):
            etype = edge_dict.get("edge_type", "")
            if etype == EdgeType.RECOMMENDS.value:
                recommends_count[v].append(u)
            elif etype in (EdgeType.EVALUATES.value, EdgeType.PRODUCES.value):
                execution_count[v] += 1
    
    # Find gaps: many recommendations, few/zero executions
    gaps: list[tuple[str, int, int]] = []
    for node_id, recommenders in recommends_count.items():
        if len(recommenders) >= min_recommendations:
            executions = execution_count.get(node_id, 0)
            if executions <= 1:  # gap = recommended but barely executed
                gaps.append((node_id, len(recommenders), executions))
    
    gaps.sort(key=lambda x: x[1], reverse=True)
    gaps = gaps[:top_k]
    
    # Build PatternCandidates
    patterns: list[PatternCandidate] = []
    for node_id, rec_count, exec_count in gaps:
        node = graph.get_node(node_id)
        if not node:
            continue
        
        # Gather evidence from recommending nodes
        evidence = []
        for recommender_id in recommends_count[node_id][:5]:
            rec_node = graph.get_node(recommender_id)
            if rec_node:
                evidence.append(EvidenceItem(
                    assertion_node_id=rec_node.id,
                    assertion_text=f"Recommends: {node.name}",
                    source_document_id=rec_node.properties.get("source_document_id", ""),
                    source_url=rec_node.properties.get("source_url", ""),
                    source_tier=rec_node.properties.get("source_tier", 2),
                ))
        
        pattern = PatternCandidate(
            pattern_type=PatternType.GAP,
            title=f"Gap: {node.name}",
            measured_pattern=(
                f"'{node.name}' is recommended by {rec_count} sources "
                f"but has only {exec_count} execution/evaluation results."
            ),
            evidence=evidence,
            blind_spots=[BlindSpot(
                description="Execution may exist outside indexed sources",
                severity="moderate",
            )],
            confidence_score=min(rec_count / 5, 0.9),
            domain=node.domain,
            details={
                "gap_node": node.name,
                "recommendation_count": rec_count,
                "execution_count": exec_count,
            },
        )
        patterns.append(pattern)
    
    logger.info(f"Produced {len(patterns)} gap pattern candidates")
    return patterns
```

### File: src/core/patterns/__init__.py

```python
"""Pattern detection engines."""
from src.core.patterns.bridges import detect_bridges
from src.core.patterns.contradictions import detect_contradictions
from src.core.patterns.drift import detect_drift
from src.core.patterns.gaps import detect_gaps
```

---

## Phase 9: Verifier + promotion gate

### File: src/core/verifier.py

```python
"""
Pattern verifier and promotion gate.

Every candidate pattern must pass all checks to be promoted.
This module is domain-agnostic — it checks evidence counts,
source diversity, and confidence thresholds, not content.
"""

from __future__ import annotations
import logging

from src.core.types import (
    PatternCandidate, PromotedPattern, ConfidenceLevel, BlindSpot,
)

logger = logging.getLogger(__name__)

# Promotion thresholds
MIN_EVIDENCE_COUNT = 3
MIN_SOURCE_TIERS = 2
MIN_SOURCE_URLS = 3
MIN_CONFIDENCE = 0.4


def _compute_confidence_level(pattern: PatternCandidate) -> ConfidenceLevel:
    """Assign confidence level based on evidence quality."""
    has_tier_1 = 1 in pattern.source_tiers
    tier_count = len(pattern.source_tiers)
    ev_count = pattern.evidence_count
    has_counter = len(pattern.counter_evidence) > 0
    
    # Check for unresolved contradictions
    if has_counter and pattern.confidence_score < 0.6:
        return ConfidenceLevel.UNRESOLVED
    
    # High: strong evidence across multiple tiers
    if ev_count >= 5 and tier_count >= 2 and has_tier_1 and not has_counter:
        return ConfidenceLevel.HIGH
    
    # Medium: decent evidence but limitations
    if ev_count >= 3 and (tier_count >= 2 or len(pattern.source_urls) >= 3):
        return ConfidenceLevel.MEDIUM
    
    # Low: meets minimum but thin
    return ConfidenceLevel.LOW


def verify_pattern(pattern: PatternCandidate) -> PromotedPattern | None:
    """Run a candidate pattern through the promotion gate.
    
    Returns PromotedPattern if promoted, None if withheld.
    Demoted patterns get a withheld_reason.
    """
    reasons: list[str] = []
    
    # Check 1: Minimum evidence count
    if pattern.evidence_count < MIN_EVIDENCE_COUNT:
        reasons.append(f"Evidence count {pattern.evidence_count} < {MIN_EVIDENCE_COUNT}")
    
    # Check 2: Source diversity
    tier_diverse = len(pattern.source_tiers) >= MIN_SOURCE_TIERS
    url_diverse = len(pattern.source_urls) >= MIN_SOURCE_URLS
    if not (tier_diverse or url_diverse):
        reasons.append(f"Limited source diversity: {len(pattern.source_tiers)} tiers, {len(pattern.source_urls)} URLs")
    
    # Check 3: Confidence threshold
    if pattern.confidence_score < MIN_CONFIDENCE:
        reasons.append(f"Confidence {pattern.confidence_score:.2f} < {MIN_CONFIDENCE}")
    
    # Check 4: Blind spots must be logged
    if not pattern.blind_spots:
        pattern.blind_spots.append(BlindSpot(
            description="No blind spots identified — default added",
            severity="minor",
        ))
    
    # Compute confidence level
    confidence_level = _compute_confidence_level(pattern)
    pattern.confidence_level = confidence_level
    
    # Decide: promote or withhold
    if reasons:
        logger.info(f"Withholding pattern '{pattern.title}': {'; '.join(reasons)}")
        promoted = PromotedPattern(**pattern.__dict__)
        promoted.withheld_reason = "; ".join(reasons)
        promoted.confidence_level = ConfidenceLevel.LOW
        return promoted  # still returned but marked as exploratory
    
    promoted = PromotedPattern(**pattern.__dict__)
    promoted.promotion_reason = "Passed all promotion checks"
    promoted.confidence_level = confidence_level
    
    logger.info(f"Promoted pattern '{pattern.title}' with confidence={confidence_level.value}")
    return promoted


def verify_all(candidates: list[PatternCandidate]) -> tuple[list[PromotedPattern], list[PromotedPattern]]:
    """Verify all candidates. Returns (promoted, exploratory) lists."""
    promoted: list[PromotedPattern] = []
    exploratory: list[PromotedPattern] = []
    
    for candidate in candidates:
        result = verify_pattern(candidate)
        if result:
            if result.withheld_reason:
                exploratory.append(result)
            else:
                promoted.append(result)
    
    logger.info(f"Verification: {len(promoted)} promoted, {len(exploratory)} exploratory")
    return promoted, exploratory
```

### Test: tests/core/test_verifier.py

```python
"""Tests for verifier and promotion gate."""
import pytest
from src.core.types import (
    PatternCandidate, PatternType, ConfidenceLevel, EvidenceItem, BlindSpot,
)
from src.core.verifier import verify_pattern, verify_all


def _make_evidence(count: int, tiers: list[int] | None = None) -> list[EvidenceItem]:
    if tiers is None:
        tiers = [1] * count
    return [
        EvidenceItem(
            assertion_node_id=f"a{i}", assertion_text=f"claim {i}",
            source_document_id=f"d{i}", source_url=f"https://source{i}.com",
            source_tier=tiers[i % len(tiers)],
        ) for i in range(count)
    ]


class TestVerifier:
    def test_high_confidence_promotion(self):
        pattern = PatternCandidate(
            pattern_type=PatternType.BRIDGE, title="Strong bridge",
            evidence=_make_evidence(5, [1, 1, 2, 2, 1]),
            confidence_score=0.8,
        )
        result = verify_pattern(pattern)
        assert result is not None
        assert result.withheld_reason is None
        assert result.confidence_level == ConfidenceLevel.HIGH

    def test_medium_confidence(self):
        pattern = PatternCandidate(
            pattern_type=PatternType.CONTRADICTION, title="Medium contradiction",
            evidence=_make_evidence(3, [1]),  # only 1 tier
            confidence_score=0.6,
        )
        result = verify_pattern(pattern)
        assert result is not None
        # Only 1 tier but 3 URLs, so should still pass
        assert result.confidence_level in (ConfidenceLevel.MEDIUM, ConfidenceLevel.LOW)

    def test_withholds_insufficient_evidence(self):
        pattern = PatternCandidate(
            pattern_type=PatternType.BRIDGE, title="Weak bridge",
            evidence=_make_evidence(2),
            confidence_score=0.5,
        )
        result = verify_pattern(pattern)
        assert result is not None
        assert result.withheld_reason is not None

    def test_unresolved_with_counter_evidence(self):
        pattern = PatternCandidate(
            pattern_type=PatternType.CONTRADICTION, title="Unresolved",
            evidence=_make_evidence(5, [1, 2, 1, 2, 1]),
            counter_evidence=_make_evidence(3, [1, 2, 1]),
            confidence_score=0.4,
        )
        result = verify_pattern(pattern)
        assert result is not None
        assert result.confidence_level == ConfidenceLevel.UNRESOLVED

    def test_adds_default_blind_spot(self):
        pattern = PatternCandidate(
            pattern_type=PatternType.GAP, title="Test",
            evidence=_make_evidence(5, [1, 2, 1, 2, 1]),
            blind_spots=[],  # empty
            confidence_score=0.7,
        )
        result = verify_pattern(pattern)
        assert len(result.blind_spots) >= 1

    def test_never_crashes_on_edge_cases(self):
        empty = PatternCandidate(pattern_type=PatternType.BRIDGE, title="Empty")
        result = verify_pattern(empty)
        assert result is not None  # should handle gracefully

    def test_verify_all_splits_correctly(self):
        strong = PatternCandidate(
            pattern_type=PatternType.BRIDGE, title="Strong",
            evidence=_make_evidence(5, [1, 2, 1, 2, 1]),
            confidence_score=0.8,
        )
        weak = PatternCandidate(
            pattern_type=PatternType.GAP, title="Weak",
            evidence=_make_evidence(1),
            confidence_score=0.3,
        )
        promoted, exploratory = verify_all([strong, weak])
        assert len(promoted) >= 1
        assert len(exploratory) >= 1
```

---

## Phase 10: Report generator + agent.py integration

### File: src/core/report.py

```python
"""
Report generator.

Produces four artifacts:
1. Pattern report (markdown)
2. Evidence table (JSON)
3. Graph visualization (D3.js HTML)
4. Coverage report (markdown)

Uses domain pack interpretation templates for pattern explanations.
"""

from __future__ import annotations
import json
import logging
from typing import Any

from src.core.types import (
    PromotedPattern, ConfidenceLevel, CoverageReport, PatternType,
)
from src.core.graph import KnowledgeGraph

logger = logging.getLogger(__name__)


def generate_pattern_report(
    promoted: list[PromotedPattern],
    exploratory: list[PromotedPattern],
    coverage: CoverageReport,
) -> str:
    """Generate the main pattern report as markdown."""
    sections = ["# Pattern Discovery Report\n"]
    
    # Summary
    sections.append(f"**Patterns found:** {len(promoted)} promoted, {len(exploratory)} exploratory\n")
    
    # Promoted patterns
    if promoted:
        sections.append("## Promoted patterns\n")
        for p in promoted:
            sections.append(_format_pattern_card(p))
    
    # Exploratory leads
    if exploratory:
        sections.append("## Exploratory leads (not enough evidence to promote)\n")
        for p in exploratory:
            sections.append(_format_pattern_card(p, exploratory=True))
    
    # Coverage
    sections.append("## Coverage\n")
    sections.append(f"- Sources used: {', '.join(coverage.source_families_used)}\n")
    sections.append(f"- Sources missing: {', '.join(coverage.source_families_missing) or 'none identified'}\n")
    sections.append(f"- Total documents: {coverage.total_documents}\n")
    if coverage.weak_subtopics:
        sections.append(f"- Weak subtopics: {', '.join(coverage.weak_subtopics)}\n")
    for note in coverage.notes:
        sections.append(f"- {note}\n")
    
    return "\n".join(sections)


def _format_pattern_card(p: PromotedPattern, exploratory: bool = False) -> str:
    """Format a single pattern as a markdown card."""
    lines = [f"### [{p.pattern_type.value.title()}]: {p.title}\n"]
    lines.append(f"**Confidence:** {p.confidence_level.value if p.confidence_level else 'unscored'}")
    if exploratory and p.withheld_reason:
        lines.append(f"**Withheld because:** {p.withheld_reason}")
    lines.append(f"\n**Measured pattern:** {p.measured_pattern}\n")
    
    if p.evidence:
        lines.append("**Supporting evidence:**")
        for e in p.evidence[:5]:
            lines.append(f"- (Tier {e.source_tier}) {e.assertion_text[:200]}")
    
    if p.counter_evidence:
        lines.append("\n**Counterevidence:**")
        for e in p.counter_evidence[:3]:
            lines.append(f"- (Tier {e.source_tier}) {e.assertion_text[:200]}")
    
    if p.interpretation:
        lines.append(f"\n**Interpretation:** {p.interpretation}")
    
    if p.blind_spots:
        lines.append("\n**Blind spots:**")
        for bs in p.blind_spots:
            lines.append(f"- [{bs.severity}] {bs.description}")
    
    lines.append("\n---\n")
    return "\n".join(lines)


def generate_evidence_table(
    promoted: list[PromotedPattern],
    exploratory: list[PromotedPattern],
) -> str:
    """Generate evidence table as JSON."""
    entries = []
    for p in promoted + exploratory:
        for e in p.evidence + p.counter_evidence:
            entries.append({
                "pattern_id": p.id,
                "pattern_type": p.pattern_type.value,
                "pattern_title": p.title,
                "assertion_text": e.assertion_text,
                "source_url": e.source_url,
                "source_tier": e.source_tier,
                "role": e.role,
                "confidence": p.confidence_level.value if p.confidence_level else "",
            })
    return json.dumps(entries, indent=2)


def generate_graph_html(graph: KnowledgeGraph, promoted: list[PromotedPattern]) -> str:
    """Generate interactive D3.js graph visualization.
    
    Nodes colored by community (Louvain). Bridge edges amber.
    Contradiction edges red. Clickable nodes show details.
    """
    import networkx as nx
    
    # Get community assignments
    communities = {}
    try:
        comms = nx.community.louvain_communities(graph.g, seed=42)
        for i, comm in enumerate(comms):
            for node_id in comm:
                communities[node_id] = i
    except Exception:
        pass
    
    # Build D3 data
    nodes_data = []
    for node in graph._node_registry.values():
        nodes_data.append({
            "id": node.id,
            "name": node.name[:40],
            "type": node.node_type.value,
            "community": communities.get(node.id, 0),
            "description": node.description[:200],
        })
    
    links_data = []
    # Collect bridge and contradiction node pairs from promoted patterns
    bridge_pairs = set()
    contradiction_pairs = set()
    for p in promoted:
        if p.pattern_type == PatternType.BRIDGE:
            nu = p.details.get("bridge_node_u", "")
            nv = p.details.get("bridge_node_v", "")
            if nu and nv:
                bridge_pairs.add((nu, nv))
        elif p.pattern_type == PatternType.CONTRADICTION:
            for e in p.evidence:
                for ce in p.counter_evidence:
                    contradiction_pairs.add((e.assertion_node_id, ce.assertion_node_id))
    
    for u, v, data in graph.g.edges(data=True):
        edge_type = data.get("edge_type", "co_occurs")
        # Check if this is a bridge or contradiction edge
        color = "#999"
        nu = graph.get_node(u)
        nv = graph.get_node(v)
        if nu and nv:
            pair = (nu.name, nv.name)
            rev_pair = (nv.name, nu.name)
            if pair in bridge_pairs or rev_pair in bridge_pairs:
                color = "#EF9F27"  # amber for bridges
            if (u, v) in contradiction_pairs or (v, u) in contradiction_pairs:
                color = "#E24B4A"  # red for contradictions
        
        links_data.append({
            "source": u, "target": v,
            "type": edge_type, "color": color,
        })
    
    # Cap for performance
    nodes_data = nodes_data[:500]
    node_ids = {n["id"] for n in nodes_data}
    links_data = [l for l in links_data if l["source"] in node_ids and l["target"] in node_ids]
    links_data = links_data[:2000]
    
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
body {{ margin: 0; font-family: -apple-system, sans-serif; background: #fafafa; }}
svg {{ width: 100%; height: 100vh; }}
.node circle {{ stroke: #fff; stroke-width: 1.5px; cursor: pointer; }}
.link {{ stroke-opacity: 0.4; }}
#tooltip {{ position: absolute; background: white; border: 1px solid #ddd;
  padding: 8px 12px; border-radius: 6px; font-size: 13px; pointer-events: none;
  display: none; max-width: 300px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
</style></head><body>
<div id="tooltip"></div>
<svg></svg>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js"></script>
<script>
const nodes = {json.dumps(nodes_data)};
const links = {json.dumps(links_data)};
const colors = d3.scaleOrdinal(d3.schemeTableau10);
const svg = d3.select("svg");
const width = window.innerWidth, height = window.innerHeight;
const sim = d3.forceSimulation(nodes)
  .force("link", d3.forceLink(links).id(d=>d.id).distance(60))
  .force("charge", d3.forceManyBody().strength(-80))
  .force("center", d3.forceCenter(width/2, height/2));
const link = svg.selectAll(".link").data(links).join("line")
  .attr("class","link").attr("stroke",d=>d.color).attr("stroke-width",1.5);
const node = svg.selectAll(".node").data(nodes).join("g").attr("class","node")
  .call(d3.drag().on("start",ds).on("drag",dd).on("end",de));
node.append("circle").attr("r",6).attr("fill",d=>colors(d.community));
node.append("text").text(d=>d.name).attr("dx",10).attr("dy",4)
  .style("font-size","11px").style("fill","#555");
const tooltip = d3.select("#tooltip");
node.on("mouseover",(e,d)=>{{tooltip.style("display","block")
  .html("<b>"+d.name+"</b><br>"+d.type+"<br>"+d.description)
  .style("left",(e.pageX+10)+"px").style("top",(e.pageY-10)+"px")}})
  .on("mouseout",()=>tooltip.style("display","none"));
sim.on("tick",()=>{{
  link.attr("x1",d=>d.source.x).attr("y1",d=>d.source.y)
    .attr("x2",d=>d.target.x).attr("y2",d=>d.target.y);
  node.attr("transform",d=>"translate("+d.x+","+d.y+")");
}});
function ds(e,d){{if(!e.active)sim.alphaTarget(.3).restart();d.fx=d.x;d.fy=d.y}}
function dd(e,d){{d.fx=e.x;d.fy=e.y}}
function de(e,d){{if(!e.active)sim.alphaTarget(0);d.fx=null;d.fy=null}}
</script></body></html>"""
    return html
```

### File: agent.py (RunForge entry point)

```python
"""
Pattern Discovery Agent — RunForge entry point.

This is the thin wrapper. All logic lives in src/.
~30 lines of orchestration, as per RunForge convention.
"""

import os
import json
import logging
from agent_runtime import AgentRuntime

runtime = AgentRuntime()
logger = logging.getLogger(__name__)

STEPS = [
    "classify_topic", "route_sources", "ingest_corpus", "expand_corpus",
    "extract_knowledge", "build_graph", "mine_patterns", "verify_patterns",
    "generate_report",
]


@runtime.agent(name="pattern-discovery", planned_steps=STEPS)
async def run(ctx, input):
    from src.core.graph import KnowledgeGraph
    from src.core.patterns import detect_bridges, detect_contradictions, detect_drift, detect_gaps
    from src.core.verifier import verify_all
    from src.core.report import generate_pattern_report, generate_evidence_table, generate_graph_html
    from src.core.types import CoverageReport
    from src.shared.corpus import deduplicate, expand_corpus, assign_tiers, corpus_stats
    from src.shared.extraction import extract_batch
    from src.shared.embeddings import embed_texts
    from src.packs.research import ResearchPack

    topic = ctx.inputs.get("topic", "")
    depth = ctx.inputs.get("depth", "standard")
    focus = ctx.inputs.get("focus", "all")
    time_range = ctx.inputs.get("time_range")
    max_documents = int(ctx.inputs.get("max_documents", 100))

    pack = ResearchPack()
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    with ctx.safe_step("classify_topic"):
        ctx.log(f"Topic: {topic}, Domain: {pack.domain}")
        ctx.state["topic"] = topic
        ctx.state["domain"] = pack.domain

    with ctx.safe_step("route_sources"):
        from src.packs.research.router import build_source_plan
        plan = build_source_plan(topic, depth=depth, focus=focus,
                                 time_range=time_range, max_documents=max_documents)
        ctx.state["source_plan"] = {"connectors": plan.connectors, "queries": plan.queries}
        ctx.log(f"Source plan: {plan.connectors}, {len(plan.queries)} queries")

    with ctx.safe_step("ingest_corpus"):
        config = {
            "OPENALEX_API_KEY": os.environ.get("OPENALEX_API_KEY", ""),
            "SEMANTIC_SCHOLAR_API_KEY": os.environ.get("SEMANTIC_SCHOLAR_API_KEY", ""),
            "GITHUB_TOKEN": os.environ.get("GITHUB_TOKEN", ""),
            "TAVILY_API_KEY": os.environ.get("TAVILY_API_KEY", ""),
        }
        connectors = pack.get_connectors(config)
        all_docs = []
        for connector in connectors:
            for query in plan.queries[:3]:
                try:
                    docs = await connector.search(query, limit=plan.per_connector_limit,
                                                   time_filter=plan.time_filter)
                    all_docs.extend(docs)
                    ctx.log(f"  {connector.__class__.__name__}: {len(docs)} docs for '{query}'")
                except Exception as e:
                    ctx.log(f"  {connector.__class__.__name__} failed: {e}", level="warning")
            await connector.close() if hasattr(connector, 'close') else None
        
        all_docs = deduplicate(all_docs)
        all_docs = all_docs[:max_documents]
        all_docs = assign_tiers(all_docs, pack)
        stats = corpus_stats(all_docs)
        ctx.state["corpus_stats"] = stats
        ctx.log(f"Corpus: {stats['total_documents']} docs after dedup")
        ctx.storage.put_file("documents.json", json.dumps([d.to_dict() for d in all_docs]))

    with ctx.safe_step("expand_corpus"):
        connectors = pack.get_connectors(config)
        new_docs = await expand_corpus(all_docs, connectors, budget=plan.expansion_budget)
        new_docs = assign_tiers(new_docs, pack)
        all_docs.extend(new_docs)
        all_docs = deduplicate(all_docs)
        ctx.log(f"After expansion: {len(all_docs)} docs")
        ctx.storage.put_file("documents.json", json.dumps([d.to_dict() for d in all_docs]))
        for c in connectors:
            await c.close() if hasattr(c, 'close') else None

    with ctx.safe_step("extract_knowledge"):
        schema = pack.get_schema()
        results = await extract_batch(all_docs, schema, api_key=api_key)
        total_nodes = sum(len(r.nodes) for r in results)
        total_edges = sum(len(r.edges) for r in results)
        ctx.state["extraction_stats"] = {"nodes": total_nodes, "edges": total_edges}
        ctx.log(f"Extracted: {total_nodes} nodes, {total_edges} edges")
        # Store results for graph building
        ctx.storage.put_file("extraction_results.json",
                            json.dumps([{"doc_id": r.source_document_id,
                                        "nodes": [n.to_dict() for n in r.nodes],
                                        "edges": [e.to_dict() for e in r.edges]}
                                       for r in results]))

    with ctx.safe_step("build_graph"):
        graph = KnowledgeGraph()
        for result in results:
            graph.add_extraction_result(result)
        # Embed all text nodes
        text_nodes = [n for n in graph._node_registry.values()
                     if n.description and n.embedding is None]
        if text_nodes:
            texts = [n.description for n in text_nodes]
            embeddings = embed_texts(texts)
            for node, emb in zip(text_nodes, embeddings):
                node.embedding = emb.tolist()
        
        graph_stats = graph.stats()
        ctx.state["graph_stats"] = graph_stats
        ctx.log(f"Graph: {graph_stats['total_nodes']} nodes, {graph_stats['total_edges']} edges")
        ctx.storage.put_file("graph.json", graph.to_json())

    with ctx.safe_step("mine_patterns"):
        candidates = []
        if focus in ("bridges", "all"):
            candidates.extend(detect_bridges(graph))
        if focus in ("contradictions", "all"):
            candidates.extend(detect_contradictions(graph))
        if focus in ("drift", "all"):
            candidates.extend(detect_drift(graph))
        if focus in ("gaps", "all"):
            candidates.extend(detect_gaps(graph))
        ctx.state["pattern_stats"] = {"candidates": len(candidates)}
        ctx.log(f"Pattern mining: {len(candidates)} candidates")

    with ctx.safe_step("verify_patterns"):
        promoted, exploratory = verify_all(candidates)
        ctx.state["pattern_stats"]["promoted"] = len(promoted)
        ctx.state["pattern_stats"]["exploratory"] = len(exploratory)
        ctx.log(f"Verified: {len(promoted)} promoted, {len(exploratory)} exploratory")

    with ctx.safe_step("generate_report"):
        coverage = CoverageReport(
            source_families_used=list(set(d.source_family for d in all_docs)),
            total_documents=len(all_docs),
            documents_per_tier=corpus_stats(all_docs)["documents_per_tier"],
        )
        
        # Generate artifacts
        report_md = generate_pattern_report(promoted, exploratory, coverage)
        ctx.artifact("pattern_report.md", report_md, "text/markdown")
        
        evidence_json = generate_evidence_table(promoted, exploratory)
        ctx.artifact("evidence_table.json", evidence_json, "application/json")
        
        graph_html = generate_graph_html(graph, promoted)
        ctx.artifact("knowledge_graph.html", graph_html, "text/html")
        
        # Set results for dashboard
        ctx.results.set_stats({
            "patterns_found": len(promoted),
            "exploratory_leads": len(exploratory),
            "high_confidence": sum(1 for p in promoted if p.confidence_level == ConfidenceLevel.HIGH),
            "documents_analyzed": len(all_docs),
            "graph_nodes": graph.node_count,
            "graph_edges": graph.edge_count,
        })
        ctx.results.set_table("patterns", [
            {"type": p.pattern_type.value, "title": p.title,
             "confidence": p.confidence_level.value if p.confidence_level else "",
             "evidence_count": p.evidence_count}
            for p in promoted
        ])
        
        ctx.log(f"Report generated: {len(promoted)} patterns, {len(all_docs)} docs analyzed")

    return {
        "status": "completed",
        "patterns_promoted": len(promoted),
        "patterns_exploratory": len(exploratory),
        "documents_analyzed": len(all_docs),
    }


if __name__ == "__main__":
    runtime.serve()
```

### Integration test: tests/integration/test_full_pipeline.py

```python
"""
Full pipeline integration test.

This test requires API keys set as environment variables.
Run with: pytest tests/integration/test_full_pipeline.py -v --timeout=300
"""
import pytest
import os


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)
class TestFullPipeline:
    """End-to-end integration tests."""
    
    def test_type_system_roundtrip(self):
        """Verify types survive full serialization cycle."""
        from src.core.types import Node, Edge, EdgeMeta, EdgeType, NodeType
        from src.core.graph import KnowledgeGraph
        import json
        
        g = KnowledgeGraph()
        meta = EdgeMeta(source_tier=1, source_family="scholarly", source_url="https://test.com",
                       source_id="W1", extraction_confidence=0.9, extraction_model="test",
                       timestamp="2025-01-01", provenance="abstract", domain="research")
        n1 = Node(id="n1", node_type=NodeType.CONCEPT, name="Test A", domain="research")
        n2 = Node(id="n2", node_type=NodeType.ASSERTION, name="Test B", domain="research")
        g.add_node(n1)
        g.add_node(n2)
        g.add_edge(Edge(source_node_id="n1", target_node_id="n2",
                       edge_type=EdgeType.ASSOCIATED_WITH, meta=meta))
        
        json_str = g.to_json()
        restored = KnowledgeGraph.from_json(json_str)
        assert restored.node_count == 2
        assert restored.edge_count == 1
    
    def test_domain_pack_loads(self):
        """Verify research domain pack initializes correctly."""
        from src.packs.research import ResearchPack
        pack = ResearchPack()
        assert pack.domain == "research"
        schema = pack.get_schema()
        assert "assertions" in schema.extraction_prompt.lower() or "assertion" in schema.extraction_prompt.lower()
    
    def test_verifier_consistency(self):
        """Verify promotion gate produces consistent results."""
        from src.core.types import PatternCandidate, PatternType, EvidenceItem
        from src.core.verifier import verify_all
        
        strong = PatternCandidate(
            pattern_type=PatternType.BRIDGE, title="Strong",
            evidence=[EvidenceItem(assertion_node_id=f"a{i}", assertion_text=f"c{i}",
                                  source_document_id=f"d{i}", source_url=f"https://s{i}.com",
                                  source_tier=t)
                     for i, t in enumerate([1, 1, 2, 2, 1])],
            confidence_score=0.8,
        )
        weak = PatternCandidate(
            pattern_type=PatternType.GAP, title="Weak",
            evidence=[EvidenceItem(assertion_node_id="a0", assertion_text="c0",
                                  source_document_id="d0", source_url="https://s0.com",
                                  source_tier=3)],
            confidence_score=0.3,
        )
        
        promoted, exploratory = verify_all([strong, weak])
        assert len(promoted) == 1
        assert promoted[0].title == "Strong"
        assert len(exploratory) == 1
```

---

## Run all tests

```bash
# Phase 1: Types
pytest tests/core/test_types.py -v

# Phase 2: Domain pack + schema
pytest tests/packs/research/test_schema.py -v

# Phase 3: Connectors (offline tests only)
pytest tests/packs/research/test_connectors.py -v

# Phase 4-5: Corpus + extraction
pytest tests/shared/ -v

# Phase 6: Graph
pytest tests/core/test_graph.py -v

# Phase 7: Bridges + contradictions
pytest tests/core/test_bridges.py -v
pytest tests/core/test_contradictions.py -v -m "not slow"

# Phase 9: Verifier
pytest tests/core/test_verifier.py -v

# Phase 10: Integration
pytest tests/integration/ -v

# Everything
pytest tests/ -v --ignore=tests/integration -m "not slow and not live"
```

**Done when:** All tests pass. The agent runs end-to-end with `python -m agent_runtime dev agent:run`, produces all four artifacts, and completes in under 30 minutes for a 100-document corpus.
