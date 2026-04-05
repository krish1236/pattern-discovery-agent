"""Real HTTP calls to public APIs. Set RUN_LIVE_CONNECTOR_TESTS=1 to run."""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.live


@pytest.mark.asyncio
async def test_openalex_search_returns_documents() -> None:
    if os.environ.get("RUN_LIVE_CONNECTOR_TESTS") != "1":
        pytest.skip("Set RUN_LIVE_CONNECTOR_TESTS=1 to run live connector tests")

    from src.packs.research.connectors.openalex import OpenAlexConnector

    connector = OpenAlexConnector()
    try:
        docs = await connector.search("machine learning", limit=3)
    finally:
        await connector.close()

    assert len(docs) >= 1
    assert docs[0].title
    assert docs[0].source_family == "scholarly"
