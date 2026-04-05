"""arXiv Atom API connector."""

from __future__ import annotations

import asyncio
import time
import xml.etree.ElementTree as ET

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.core.types import SourceDocument
from src.domain_pack import SourceConnector

ARXIV_BASE = "https://export.arxiv.org/api"
NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


def _parse_entry(entry: ET.Element, domain: str = "research") -> SourceDocument:
    title = (entry.findtext("atom:title", "", NS) or "").strip().replace("\n", " ")
    abstract = (entry.findtext("atom:summary", "", NS) or "").strip().replace("\n", " ")

    raw_id = entry.findtext("atom:id", "", NS) or ""
    arxiv_id = raw_id.split("/abs/")[-1] if "/abs/" in raw_id else raw_id.split("/")[-1]

    published_el = entry.findtext("atom:published", "", NS)
    published = published_el[:10] if published_el else None

    authors: list[str] = []
    for author in entry.findall("atom:author", NS)[:10]:
        name = author.findtext("atom:name", "", NS)
        if name:
            authors.append(name)

    categories: list[str] = []
    pc = entry.find("arxiv:primary_category", NS)
    if pc is not None:
        term = pc.get("term", "")
        if term:
            categories.append(term)
    for cat in entry.findall("atom:category", NS):
        term = cat.get("term", "")
        if term and term not in categories:
            categories.append(term)

    return SourceDocument(
        source_id=arxiv_id,
        source_family="scholarly",
        source_url=f"https://arxiv.org/abs/{arxiv_id}",
        title=title,
        abstract=abstract,
        authors=authors,
        publication_date=published,
        source_tier=2,
        domain=domain,
        metadata={
            "arxiv_id": arxiv_id,
            "categories": categories,
            "source_type": "preprint",
        },
    )


class ArxivConnector(SourceConnector):
    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None
        self._last_request_time: float = 0.0

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0, follow_redirects=True)
        return self._client

    async def _rate_limit(self) -> None:
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < 3.0:
            await asyncio.sleep(3.0 - elapsed)
        self._last_request_time = time.time()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=5, max=30),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.ConnectError, httpx.TimeoutException)),
    )
    async def search(self, query: str, limit: int = 20, **kwargs: object) -> list[SourceDocument]:
        await self._rate_limit()
        client = await self._get_client()
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": min(limit, 50),
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        resp = await client.get(f"{ARXIV_BASE}/query", params=params)
        resp.raise_for_status()

        root = ET.fromstring(resp.text)
        results: list[SourceDocument] = []
        for entry in root.findall("atom:entry", NS):
            doc = _parse_entry(entry)
            if doc.title and doc.title != "Error":
                results.append(doc)
        return results

    async def get(self, source_id: str) -> SourceDocument | None:
        await self._rate_limit()
        client = await self._get_client()
        params = {"id_list": source_id, "max_results": 1}
        resp = await client.get(f"{ARXIV_BASE}/query", params=params)
        resp.raise_for_status()

        root = ET.fromstring(resp.text)
        entries = root.findall("atom:entry", NS)
        if entries:
            doc = _parse_entry(entries[0])
            if doc.title and doc.title != "Error":
                return doc
        return None

    async def expand(self, source_id: str, limit: int = 10) -> list[SourceDocument]:
        return []

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
