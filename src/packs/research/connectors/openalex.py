"""OpenAlex API connector."""

from __future__ import annotations

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.core.types import SourceDocument
from src.domain_pack import SourceConnector

OPENALEX_BASE = "https://api.openalex.org"


def _invert_abstract(inverted: dict | None) -> str:
    if not inverted:
        return ""
    word_positions: list[tuple[int, str]] = []
    for word, positions in inverted.items():
        for pos in positions:
            word_positions.append((pos, word))
    word_positions.sort()
    return " ".join(w for _, w in word_positions)


def _parse_work(work: dict, domain: str = "research") -> SourceDocument:
    openalex_id = work.get("id", "").replace("https://openalex.org/", "")
    work_type = work.get("type", "")
    source_type = ""
    if work_type in ("journal-article", "proceedings-article"):
        source_type = "journal"

    authors: list[str] = []
    for authorship in work.get("authorships", [])[:10]:
        author_name = authorship.get("author", {}).get("display_name", "")
        if author_name:
            authors.append(author_name)

    abstract = _invert_abstract(work.get("abstract_inverted_index"))
    doi = work.get("doi", "") or ""
    if isinstance(doi, str) and doi and not doi.startswith("http"):
        doi = f"https://doi.org/{doi}"

    return SourceDocument(
        source_id=openalex_id,
        source_family="scholarly",
        source_url=doi or work.get("id", ""),
        title=work.get("title", "") or "",
        abstract=abstract,
        authors=authors,
        publication_date=work.get("publication_date"),
        source_tier=1 if source_type == "journal" else 2,
        domain=domain,
        metadata={
            "openalex_id": openalex_id,
            "cited_by_count": work.get("cited_by_count", 0),
            "type": work_type,
            "source_type": source_type,
            "is_peer_reviewed": source_type == "journal",
            "concepts": [c.get("display_name", "") for c in work.get("concepts", [])[:5]],
            "is_oa": work.get("is_oa", False),
        },
    )


class OpenAlexConnector(SourceConnector):
    def __init__(self, api_key: str = "", email: str = ""):
        self.api_key = api_key
        self.email = email
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            headers: dict[str, str] = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            if self.email:
                headers["User-Agent"] = f"mailto:{self.email}"
            self._client = httpx.AsyncClient(
                base_url=OPENALEX_BASE,
                headers=headers,
                timeout=30.0,
            )
        return self._client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.ConnectError, httpx.TimeoutException)),
    )
    async def search(self, query: str, limit: int = 20, **kwargs: object) -> list[SourceDocument]:
        client = await self._get_client()
        params: dict[str, str | int] = {
            "search": query,
            "per_page": min(limit, 50),
            "sort": "cited_by_count:desc",
        }
        time_filter = kwargs.get("time_filter")
        if isinstance(time_filter, str) and "-" in time_filter:
            start, end = time_filter.split("-", 1)
            params["filter"] = f"publication_year:{start.strip()}-{end.strip()}"

        resp = await client.get("/works", params=params)
        resp.raise_for_status()
        data = resp.json()

        results: list[SourceDocument] = []
        for work in data.get("results", []):
            doc = _parse_work(work)
            if doc.title:
                results.append(doc)
        return results

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.ConnectError, httpx.TimeoutException)),
    )
    async def get(self, source_id: str) -> SourceDocument | None:
        client = await self._get_client()
        try:
            resp = await client.get(f"/works/{source_id}")
            resp.raise_for_status()
            return _parse_work(resp.json())
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.ConnectError, httpx.TimeoutException)),
    )
    async def expand(self, source_id: str, limit: int = 10) -> list[SourceDocument]:
        client = await self._get_client()
        params = {
            "filter": f"cites:{source_id}",
            "per_page": min(limit, 50),
            "sort": "cited_by_count:desc",
        }
        resp = await client.get("/works", params=params)
        resp.raise_for_status()
        data = resp.json()

        results: list[SourceDocument] = []
        for work in data.get("results", []):
            doc = _parse_work(work)
            if doc.title:
                results.append(doc)
        return results

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
