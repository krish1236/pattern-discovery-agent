"""Web search via Tavily API."""

from __future__ import annotations

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.core.types import SourceDocument
from src.domain_pack import SourceConnector

TAVILY_URL = "https://api.tavily.com/search"


class WebSearchConnector(SourceConnector):
    def __init__(self, api_key: str = ""):
        self.api_key = api_key

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.ConnectError, httpx.TimeoutException)),
    )
    async def search(self, query: str, limit: int = 10, **kwargs: object) -> list[SourceDocument]:
        if not self.api_key:
            return []

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                TAVILY_URL,
                json={
                    "api_key": self.api_key,
                    "query": query,
                    "max_results": min(limit, 20),
                    "include_answer": False,
                    "search_depth": "advanced",
                },
            )
            resp.raise_for_status()
            data = resp.json()

        results: list[SourceDocument] = []
        for item in data.get("results", []):
            doc = SourceDocument(
                source_id=item.get("url", ""),
                source_family="web",
                source_url=item.get("url", ""),
                title=item.get("title", ""),
                abstract=(item.get("content", "") or "")[:2000],
                publication_date=None,
                source_tier=3,
                domain="research",
                metadata={"score": item.get("score", 0), "source_type": "web_article"},
            )
            if doc.title:
                results.append(doc)
        return results

    async def get(self, source_id: str) -> SourceDocument | None:
        return None

    async def expand(self, source_id: str, limit: int = 10) -> list[SourceDocument]:
        return []
