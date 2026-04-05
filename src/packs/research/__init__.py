"""Research / technical domain pack."""

from __future__ import annotations

from typing import Any

from src.core.types import PatternCandidate, SourceDocument
from src.domain_pack import DomainPack, DomainSchema, SourceConnector
from src.packs.research.interpret_format import format_research_interpretation
from src.packs.research.router import SourcePlan, build_source_plan, is_research_topic
from src.packs.research.schema import get_research_schema


class ResearchPack(DomainPack):
    @property
    def domain(self) -> str:
        return "research"

    def get_schema(self) -> DomainSchema:
        return get_research_schema()

    def get_connectors(self, config: dict[str, Any]) -> list[SourceConnector]:
        from src.packs.research.connectors.arxiv import ArxivConnector
        from src.packs.research.connectors.github import GitHubConnector
        from src.packs.research.connectors.openalex import OpenAlexConnector
        from src.packs.research.connectors.semantic_scholar import SemanticScholarConnector
        from src.packs.research.connectors.web_search import WebSearchConnector

        connectors: list[SourceConnector] = [
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
        is_peer_reviewed = doc.metadata.get("is_peer_reviewed", False)
        source_type = doc.metadata.get("source_type", "")
        if is_peer_reviewed or source_type == "journal":
            return 1
        if doc.source_family == "scholarly":
            return 2
        if doc.source_family == "code":
            return 2
        if doc.source_family == "web":
            return 3
        if doc.source_family == "social":
            return 4
        return 3

    def interpret(self, pattern: PatternCandidate) -> str:
        schema = self.get_schema()
        tpl = schema.interpretation_templates.get(pattern.pattern_type.value, "")
        return format_research_interpretation(pattern, tpl)


__all__ = [
    "ResearchPack",
    "SourcePlan",
    "build_source_plan",
    "get_research_schema",
    "is_research_topic",
]
