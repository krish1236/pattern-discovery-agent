"""Optional integration checks (no live APIs by default)."""

import pytest

from src.core.graph import KnowledgeGraph
from src.core.types import Edge, EdgeMeta, EdgeType, Node, NodeType
from src.packs.research import ResearchPack


@pytest.mark.integration
def test_graph_roundtrip_and_pack_loads() -> None:
    g = KnowledgeGraph()
    meta = EdgeMeta(
        source_tier=1,
        source_family="scholarly",
        source_url="https://test.com",
        source_id="W1",
        extraction_confidence=0.9,
        extraction_model="test",
        timestamp="2025-01-01",
        provenance="abstract",
        domain="research",
    )
    n1 = Node(id="n1", node_type=NodeType.CONCEPT, name="Test A", domain="research")
    n2 = Node(id="n2", node_type=NodeType.ASSERTION, name="Test B", domain="research")
    g.add_node(n1)
    g.add_node(n2)
    g.add_edge(
        Edge(
            source_node_id="n1",
            target_node_id="n2",
            edge_type=EdgeType.ASSOCIATED_WITH,
            meta=meta,
        )
    )
    restored = KnowledgeGraph.from_json(g.to_json())
    assert restored.node_count == 2
    assert restored.edge_count == 1

    pack = ResearchPack()
    assert pack.domain == "research"
    assert "assertions" in pack.get_schema().extraction_prompt.lower()
