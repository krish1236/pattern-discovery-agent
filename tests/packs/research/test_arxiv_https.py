"""arXiv connector URL configuration."""

from src.packs.research.connectors.arxiv import ARXIV_BASE


def test_arxiv_uses_https_base() -> None:
    assert ARXIV_BASE.startswith("https://")
    assert "export.arxiv.org" in ARXIV_BASE
