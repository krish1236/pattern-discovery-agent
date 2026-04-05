"""Load a prior run from blob storage keys written by the agent."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from src.core.graph import KnowledgeGraph
from src.core.types import Edge, ExtractionResult, Node, SourceDocument

logger = logging.getLogger(__name__)


def load_checkpoint_from_blobs(
    get_file: Callable[[str], bytes | None],
    *,
    resume: bool,
) -> tuple[list[SourceDocument], list[ExtractionResult], KnowledgeGraph] | None:
    """If ``resume`` is true and all three blobs exist, parse and return them.

    ``get_file`` is typically ``ctx.storage.get_file``. Returns ``None`` if
    anything is missing or invalid (caller runs the full pipeline).
    """
    if not resume:
        return None
    raw_docs = get_file("documents.json")
    raw_ext = get_file("extraction_results.json")
    raw_graph = get_file("graph.json")
    if not raw_docs or not raw_ext or not raw_graph:
        return None
    try:
        doc_list = json.loads(raw_docs.decode("utf-8"))
        all_docs = [SourceDocument.from_dict(x) for x in doc_list]

        ext_list = json.loads(raw_ext.decode("utf-8"))
        results: list[ExtractionResult] = []
        for item in ext_list:
            results.append(
                ExtractionResult(
                    source_document_id=item["doc_id"],
                    nodes=[Node.from_dict(n) for n in item.get("nodes", [])],
                    edges=[Edge.from_dict(e) for e in item.get("edges", [])],
                )
            )

        graph = KnowledgeGraph.from_json(raw_graph.decode("utf-8"))
        return all_docs, results, graph
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        logger.warning("Checkpoint load failed: %s", e)
        return None
