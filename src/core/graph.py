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

from src.core.types import Edge, EdgeType, ExtractionResult, Node, NodeType

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
            if similarity > 0.85:
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

    def to_json(self) -> str:
        data: dict[str, Any] = {
            "nodes": [node.to_dict() for node in self._node_registry.values()],
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
