"""Semantic Scholar Academic Graph API connector."""

from __future__ import annotations

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.core.types import SourceDocument
from src.domain_pack import SourceConnector

S2_BASE = "https://api.semanticscholar.org/graph/v1"

FIELDS = (
    "title,abstract,year,citationCount,influentialCitationCount,tldr,authors,"
    "externalIds,publicationDate,embedding.specter_v2"
)


def _parse_paper(paper: dict, domain: str = "research") -> SourceDocument:
    s2_id = paper.get("paperId", "")

    authors: list[str] = []
    for author in paper.get("authors", [])[:10]:
        name = author.get("name", "")
        if name:
            authors.append(name)

    embedding = None
    emb_data = paper.get("embedding")
    if emb_data and isinstance(emb_data, dict):
        embedding = emb_data.get("vector")

    external_ids = paper.get("externalIds") or {}
    doi = external_ids.get("DOI", "")
    doi_url = f"https://doi.org/{doi}" if doi else ""

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
        source_tier=2,
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
    def __init__(self, api_key: str = ""):
        self.api_key = api_key
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            headers: dict[str, str] = {}
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
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.ConnectError, httpx.TimeoutException)),
    )
    async def search(self, query: str, limit: int = 20, **kwargs: object) -> list[SourceDocument]:
        client = await self._get_client()
        params: dict[str, str | int] = {
            "query": query,
            "limit": min(limit, 100),
            "fields": FIELDS,
        }
        year_filter = kwargs.get("time_filter")
        if isinstance(year_filter, str) and "-" in year_filter:
            start, end = year_filter.split("-", 1)
            params["year"] = f"{start.strip()}-{end.strip()}"

        resp = await client.get("/paper/search", params=params)
        resp.raise_for_status()
        data = resp.json()

        results: list[SourceDocument] = []
        for paper in data.get("data", []):
            doc = _parse_paper(paper)
            if doc.title:
                results.append(doc)
        return results

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=3, max=15),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.ConnectError, httpx.TimeoutException)),
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
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.ConnectError, httpx.TimeoutException)),
    )
    async def expand(self, source_id: str, limit: int = 10) -> list[SourceDocument]:
        client = await self._get_client()
        resp = await client.get(
            f"/paper/{source_id}/citations",
            params={
                "fields": "title,abstract,year,citationCount,authors,externalIds,publicationDate",
                "limit": min(limit, 100),
            },
        )
        resp.raise_for_status()
        data = resp.json()

        results: list[SourceDocument] = []
        for item in data.get("data", []):
            citing_paper = item.get("citingPaper", {})
            if citing_paper and citing_paper.get("title"):
                results.append(_parse_paper(citing_paper))
        return results

    async def batch_get(self, paper_ids: list[str]) -> list[SourceDocument]:
        client = await self._get_client()
        resp = await client.post(
            "/paper/batch",
            params={"fields": FIELDS},
            json={"ids": paper_ids[:500]},
        )
        resp.raise_for_status()
        results: list[SourceDocument] = []
        for paper in resp.json():
            if paper and isinstance(paper, dict) and paper.get("title"):
                results.append(_parse_paper(paper))
        return results

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
