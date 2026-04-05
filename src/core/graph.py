"""
Knowledge graph builder.

Constructs a NetworkX graph from ExtractionResults. Handles entity
resolution (merging duplicate entities), serialization to/from JSON,
and graph statistics.

This module is domain-agnostic: it only uses Node, Edge, and EdgeMeta.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any

import networkx as nx
import numpy as np

from src.core.types import Edge, EdgeType, ExtractionResult, Node, NodeType
from src.shared.embeddings import cosine_similarity_matrix

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """Typed knowledge graph backed by NetworkX with per-edge provenance."""

    def __init__(self) -> None:
        self.g: nx.Graph = nx.Graph()
        self._node_registry: dict[str, Node] = {}
        self._name_index: dict[str, list[str]] = defaultdict(list)

    @property
    def node_count(self) -> int:
        return self.g.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self.g.number_of_edges()

    def add_node(self, node: Node) -> str:
        resolved_id = self._resolve_entity(node)
        if resolved_id and resolved_id != node.id:
            self._merge_node(resolved_id, node)
            return resolved_id

        self._node_registry[node.id] = node
        self.g.add_node(node.id, **node.to_dict())

        normalized = self._normalize_name(node.name)
        if normalized:
            self._name_index[normalized].append(node.id)

        return node.id

    def add_edge(self, edge: Edge) -> None:
        if edge.source_node_id not in self.g:
            logger.warning("Source node %s not in graph, skipping edge", edge.source_node_id)
            return
        if edge.target_node_id not in self.g:
            logger.warning("Target node %s not in graph, skipping edge", edge.target_node_id)
            return

        u, v = edge.source_node_id, edge.target_node_id
        if self.g.has_edge(u, v):
            existing = self.g.edges[u, v]
            edges_list = existing.get("edges", [])
            edges_list.append(edge.to_dict())
            self.g.edges[u, v]["edges"] = edges_list
        else:
            self.g.add_edge(
                u,
                v,
                edge_type=edge.edge_type.value,
                edges=[edge.to_dict()],
            )

    def add_extraction_result(self, result: ExtractionResult) -> dict[str, str]:
        id_mapping: dict[str, str] = {}

        for node in result.nodes:
            resolved_id = self.add_node(node)
            id_mapping[node.id] = resolved_id

        for edge in result.edges:
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
        return self._node_registry.get(node_id)

    def get_nodes_by_type(self, node_type: NodeType) -> list[Node]:
        return [n for n in self._node_registry.values() if n.node_type == node_type]

    def get_edges_by_type(self, edge_type: EdgeType) -> list[Edge]:
        result: list[Edge] = []
        for _u, _v, data in self.g.edges(data=True):
            for edge_dict in data.get("edges", []):
                if edge_dict.get("edge_type") == edge_type.value:
                    result.append(Edge.from_dict(edge_dict))
        return result

    def get_neighbors(self, node_id: str) -> list[Node]:
        if node_id not in self.g:
            return []
        return [
            self._node_registry[n]
            for n in self.g.neighbors(node_id)
            if n in self._node_registry
        ]

    def _normalize_name(self, name: str) -> str:
        return name.lower().strip().replace("-", " ").replace("_", " ")

    def _resolve_entity(self, node: Node) -> str | None:
        if node.node_type == NodeType.SOURCE_DOCUMENT:
            return None

        normalized = self._normalize_name(node.name)
        if not normalized:
            return None

        candidates = self._name_index.get(normalized, [])
        for cid in candidates:
            existing = self._node_registry.get(cid)
            if existing and existing.node_type == node.node_type:
                return cid

        for existing_name, existing_ids in self._name_index.items():
            similarity = SequenceMatcher(None, normalized, existing_name).ratio()
            if similarity > 0.75:
                for eid in existing_ids:
                    existing = self._node_registry.get(eid)
                    if existing and existing.node_type == node.node_type:
                        return eid

        return None

    def _merge_node(self, existing_id: str, new_node: Node) -> None:
        existing = self._node_registry[existing_id]
        if len(new_node.description) > len(existing.description):
            existing.description = new_node.description
        existing.properties.update(new_node.properties)
        if new_node.embedding is not None:
            if existing.embedding is None or len(new_node.embedding) > len(existing.embedding or []):
                existing.embedding = new_node.embedding

    def _purge_node_id_from_name_index(self, node_id: str) -> None:
        for key in list(self._name_index.keys()):
            ids = self._name_index[key]
            if node_id not in ids:
                continue
            kept = [x for x in ids if x != node_id]
            if kept:
                self._name_index[key] = kept
            else:
                del self._name_index[key]

    def _contract_node_into(self, keep_id: str, drop_id: str) -> None:
        """Merge drop into keep: rewire edges, drop node from graph and registry."""
        if keep_id == drop_id or drop_id not in self.g:
            return
        drop_node = self._node_registry.get(drop_id)
        if drop_node is None:
            return
        self._merge_node(keep_id, drop_node)
        self._purge_node_id_from_name_index(drop_id)

        neighbors = list(self.g.neighbors(drop_id))
        for nb in neighbors:
            edata = self.g.get_edge_data(drop_id, nb) or {}
            bundles = list(edata.get("edges", []))
            self.g.remove_edge(drop_id, nb)
            for ed in bundles:
                d = dict(ed)
                if d.get("source_node_id") == drop_id:
                    d["source_node_id"] = keep_id
                if d.get("target_node_id") == drop_id:
                    d["target_node_id"] = keep_id
                if d.get("source_node_id") == d.get("target_node_id"):
                    continue
                try:
                    self.add_edge(Edge.from_dict(d))
                except (KeyError, TypeError, ValueError) as e:
                    logger.warning(
                        "Skipping malformed edge when contracting %s into %s: %s",
                        drop_id,
                        keep_id,
                        e,
                    )

        self.g.remove_node(drop_id)
        del self._node_registry[drop_id]
        if keep_id in self.g:
            self.g.nodes[keep_id].update(self._node_registry[keep_id].to_dict())

    def merge_nodes_by_embedding(self, min_cosine: float = 0.85) -> int:
        """Post-pass: merge same-type CONCEPT/ARTIFACT pairs with high embedding similarity."""
        merge_types = {NodeType.CONCEPT, NodeType.ARTIFACT}
        eligible = [n for n in self._node_registry.values() if n.node_type in merge_types and n.embedding]
        if len(eligible) < 2:
            return 0

        ids = [n.id for n in eligible]
        embs = np.stack([np.asarray(n.embedding, dtype=np.float64) for n in eligible], axis=0)
        sim = cosine_similarity_matrix(embs)
        n = len(ids)
        parent = list(range(n))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i: int, j: int) -> None:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[max(ri, rj)] = min(ri, rj)

        for i in range(n):
            for j in range(i + 1, n):
                if sim[i, j] >= min_cosine:
                    union(i, j)

        groups: dict[int, list[int]] = defaultdict(list)
        for i in range(n):
            groups[find(i)].append(i)

        merges = 0
        for idxs in groups.values():
            if len(idxs) < 2:
                continue
            members = sorted({ids[k] for k in idxs})
            keep = members[0]
            for drop in members[1:]:
                if drop not in self._node_registry or keep not in self._node_registry:
                    continue
                self._contract_node_into(keep, drop)
                merges += 1
        return merges

    def to_json(self, *, include_embeddings: bool = False) -> str:
        data: dict[str, Any] = {
            "nodes": [
                node.to_dict(include_embedding=include_embeddings)
                for node in self._node_registry.values()
            ],
            "edges": [],
        }
        for _u, _v, edge_data in self.g.edges(data=True):
            for edge_dict in edge_data.get("edges", []):
                data["edges"].append(edge_dict)
        return json.dumps(data)

    @classmethod
    def from_json(cls, json_str: str) -> KnowledgeGraph:
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

    def stats(self) -> dict[str, Any]:
        type_counts: dict[str, int] = defaultdict(int)
        for node in self._node_registry.values():
            type_counts[node.node_type.value] += 1

        edge_type_counts: dict[str, int] = defaultdict(int)
        for _, _, edata in self.g.edges(data=True):
            for edge_dict in edata.get("edges", []):
                edge_type_counts[edge_dict.get("edge_type", "unknown")] += 1

        components = nx.number_connected_components(self.g) if self.node_count > 0 else 0

        return {
            "total_nodes": self.node_count,
            "total_edges": self.edge_count,
            "nodes_by_type": dict(type_counts),
            "edges_by_type": dict(edge_type_counts),
            "connected_components": components,
        }
