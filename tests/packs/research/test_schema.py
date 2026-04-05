"""Research domain pack tests."""

from src.core.types import EdgeType, NodeType, PatternCandidate, PatternType, SourceDocument
from src.packs.research import ResearchPack
from src.packs.research.router import build_source_plan, is_research_topic
from src.packs.research.schema import get_research_schema


class TestResearchSchema:
    def test_entity_types_map_to_valid_node_types(self) -> None:
        schema = get_research_schema()
        valid_values = {nt.value for nt in NodeType}
        for _label, universal_type in schema.entity_types.items():
            assert universal_type in valid_values, f"invalid type {universal_type}"

    def test_edge_types_map_to_valid_edge_types(self) -> None:
        schema = get_research_schema()
        valid_values = {et.value for et in EdgeType}
        for _label, universal_type in schema.edge_types.items():
            assert universal_type in valid_values, f"invalid type {universal_type}"

    def test_extraction_prompt_contains_required_placeholders(self) -> None:
        schema = get_research_schema()
        assert "{title}" in schema.extraction_prompt
        assert "{abstract}" in schema.extraction_prompt

    def test_interpretation_templates_exist_for_all_pattern_types(self) -> None:
        schema = get_research_schema()
        for pt in PatternType:
            assert pt.value in schema.interpretation_templates, f"Missing template for {pt.value}"

    def test_tier_rules_cover_key_source_types(self) -> None:
        schema = get_research_schema()
        assert schema.tier_rules["peer_reviewed"] == 1
        assert schema.tier_rules["arxiv_preprint"] == 2
        assert schema.tier_rules["vendor_blog"] == 3
        assert schema.tier_rules["forum_post"] == 4


class TestResearchRouter:
    def test_classifies_research_topics(self) -> None:
        assert is_research_topic("transformer architectures for NLP") is True
        assert is_research_topic("reinforcement learning in robotics") is True
        assert is_research_topic("AI agent frameworks 2024-2026") is True

    def test_source_plan_includes_primary_connectors(self) -> None:
        plan = build_source_plan("transformer architectures")
        assert "openalex" in plan.connectors
        assert "semantic_scholar" in plan.connectors

    def test_source_plan_depth_affects_limits(self) -> None:
        quick = build_source_plan("test", depth="quick")
        deep = build_source_plan("test", depth="deep")
        assert quick.max_documents < deep.max_documents

    def test_source_plan_adds_github_for_code_topics(self) -> None:
        plan = build_source_plan("Python web framework comparison")
        assert "github" in plan.connectors

    def test_source_plan_generates_multiple_queries(self) -> None:
        plan = build_source_plan("AI agents")
        assert len(plan.queries) >= 2


def test_openalex_connector_receives_mailto_from_config() -> None:
    pack = ResearchPack()
    connectors = pack.get_connectors({"OPENALEX_MAILTO": "dev@example.org"})
    oa = connectors[0]
    assert oa.__class__.__name__ == "OpenAlexConnector"
    assert oa.email == "dev@example.org"


class TestResearchPack:
    def test_domain_is_research(self) -> None:
        pack = ResearchPack()
        assert pack.domain == "research"

    def test_classify_tier_peer_reviewed(self) -> None:
        pack = ResearchPack()
        doc = SourceDocument(
            title="Test",
            source_family="scholarly",
            metadata={"is_peer_reviewed": True},
        )
        assert pack.classify_tier(doc) == 1

    def test_classify_tier_arxiv(self) -> None:
        pack = ResearchPack()
        doc = SourceDocument(title="Test", source_family="scholarly")
        assert pack.classify_tier(doc) == 2

    def test_classify_tier_blog(self) -> None:
        pack = ResearchPack()
        doc = SourceDocument(title="Test", source_family="web")
        assert pack.classify_tier(doc) == 3

    def test_interpret_returns_template(self) -> None:
        pack = ResearchPack()
        pattern = PatternCandidate(pattern_type=PatternType.BRIDGE)
        result = pack.interpret(pattern)
        assert len(result) > 0
        assert "communities" in result.lower() or "community" in result.lower()
