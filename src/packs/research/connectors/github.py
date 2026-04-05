"""GitHub REST API connector (repository search)."""

from __future__ import annotations

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.core.types import SourceDocument
from src.domain_pack import SourceConnector

GITHUB_API = "https://api.github.com"


def _parse_repo(repo: dict, domain: str = "research") -> SourceDocument:
    created = repo.get("created_at", "") or ""
    created_date = created[:10] if created else None
    owner = repo.get("owner") or {}
    return SourceDocument(
        source_id=str(repo.get("id", "")),
        source_family="code",
        source_url=repo.get("html_url", ""),
        title=repo.get("full_name", ""),
        abstract=repo.get("description", "") or "",
        authors=[owner.get("login", "")] if owner.get("login") else [],
        publication_date=created_date,
        source_tier=2,
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
                base_url=GITHUB_API,
                headers=headers,
                timeout=30.0,
            )
        return self._client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.ConnectError, httpx.TimeoutException)),
    )
    async def search(self, query: str, limit: int = 10, **kwargs: object) -> list[SourceDocument]:
        client = await self._get_client()
        params = {
            "q": query,
            "sort": "stars",
            "order": "desc",
            "per_page": min(limit, 30),
        }
        resp = await client.get("/search/repositories", params=params)
        resp.raise_for_status()
        results: list[SourceDocument] = []
        for repo in resp.json().get("items", []):
            doc = _parse_repo(repo)
            if doc.title:
                results.append(doc)
        return results

    async def get(self, source_id: str) -> SourceDocument | None:
        client = await self._get_client()
        try:
            resp = await client.get(f"/repos/{source_id}")
            resp.raise_for_status()
            return _parse_repo(resp.json())
        except httpx.HTTPStatusError:
            return None

    async def expand(self, source_id: str, limit: int = 10) -> list[SourceDocument]:
        return []

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
