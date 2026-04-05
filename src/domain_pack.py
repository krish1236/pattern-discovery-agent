"""
Domain pack base class and interfaces.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from src.core.types import PatternCandidate, SourceDocument


class SourceConnector(ABC):
    @abstractmethod
    async def search(self, query: str, limit: int = 20, **kwargs: Any) -> list[SourceDocument]:
        ...

    @abstractmethod
    async def get(self, source_id: str) -> SourceDocument | None:
        ...

    @abstractmethod
    async def expand(self, source_id: str, limit: int = 10) -> list[SourceDocument]:
        ...


@dataclass
class DomainSchema:
    domain: str
    entity_types: dict[str, str]
    edge_types: dict[str, str]
    tier_rules: dict[str, int]
    extraction_prompt: str
    interpretation_templates: dict[str, str]


class DomainPack(ABC):
    @property
    @abstractmethod
    def domain(self) -> str:
        ...

    @abstractmethod
    def get_schema(self) -> DomainSchema:
        ...

    @abstractmethod
    def get_connectors(self, config: dict[str, Any]) -> list[SourceConnector]:
        ...

    @abstractmethod
    def classify_tier(self, doc: SourceDocument) -> int:
        ...

    @abstractmethod
    def interpret(self, pattern: PatternCandidate) -> str:
        ...
