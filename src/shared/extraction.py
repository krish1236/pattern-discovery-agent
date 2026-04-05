"""LLM batch extraction into universal graph nodes and edges."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import anthropic

from src.core.types import Edge, EdgeMeta, EdgeType, ExtractionResult, Node, NodeType, SourceDocument
from src.domain_pack import DomainSchema

logger = logging.getLogger(__name__)

EXTRACTION_MODEL = "claude-3-5-haiku-20241022"
MAX_BATCH_SIZE = 5


def _build_batch_prompt(schema: DomainSchema, documents: list[SourceDocument]) -> str:
    doc_blocks = []
    for i, doc in enumerate(documents):
        doc_blocks.append(
            f"--- DOCUMENT {i + 1} ---\nTitle: {doc.title}\n"
            f"Abstract: {doc.abstract or '(no abstract)'}\n"
        )

    prefix = schema.extraction_prompt.split("Text to extract from:")[0]
    return (
        "You are extracting structured knowledge from multiple documents.\n"
        "For EACH document, extract assertions, entities, and relationships.\n\n"
        f"{prefix}"
        "\nProcess ALL documents below. Return a JSON array with one object per document.\n"
        "Each object has the same format: {assertions, entities, relationships}.\n"
        "Respond ONLY with a valid JSON array. No markdown, no preamble.\n\n"
        + "\n".join(doc_blocks)
    )


def _map_entity_type(domain_type: str, schema: DomainSchema) -> NodeType:
    universal = schema.entity_types.get(domain_type, "")
    try:
        return NodeType(universal)
    except ValueError:
        type_map = {
            "actor": NodeType.ACTOR,
            "concept": NodeType.CONCEPT,
            "artifact": NodeType.ARTIFACT,
            "metric": NodeType.METRIC,
        }
        return type_map.get(domain_type, NodeType.CONCEPT)


def _map_edge_type(domain_type: str, schema: DomainSchema) -> EdgeType:
    universal = schema.edge_types.get(domain_type, "")
    try:
        return EdgeType(universal)
    except ValueError:
        try:
            return EdgeType(domain_type)
        except ValueError:
            return EdgeType.CO_OCCURS


def _parse_extraction(
    raw: dict[str, Any],
    doc: SourceDocument,
    schema: DomainSchema,
) -> ExtractionResult:
    nodes: list[Node] = []
    edges: list[Edge] = []
    node_name_to_id: dict[str, str] = {}

    doc_node = Node(
        id=doc.id,
        node_type=NodeType.SOURCE_DOCUMENT,
        name=doc.title,
        description=(doc.abstract[:500] if doc.abstract else ""),
        domain=doc.domain,
        properties={
            "source_url": doc.source_url,
            "publication_date": doc.publication_date or "",
            "source_tier": doc.source_tier,
        },
    )
    nodes.append(doc_node)

    base_meta = EdgeMeta(
        source_tier=doc.source_tier,
        source_family=doc.source_family,
        source_url=doc.source_url,
        source_id=doc.source_id,
        extraction_confidence=0.8,
        extraction_model=EXTRACTION_MODEL,
        timestamp=doc.publication_date or "",
        provenance="abstract",
        domain=doc.domain,
    )

    for assertion in raw.get("assertions", []):
        text = (assertion.get("text") or "").strip()
        if not text:
            continue
        node = Node(
            node_type=NodeType.ASSERTION,
            name=text[:100],
            description=text,
            domain=doc.domain,
            properties={
                "conditions": assertion.get("conditions", ""),
                "polarity": assertion.get("polarity", "neutral"),
                "full_text": text,
                "source_document_id": doc.id,
                "source_url": doc.source_url,
                "source_tier": doc.source_tier,
                "publication_date": doc.publication_date or "",
            },
        )
        nodes.append(node)
        node_name_to_id[text[:100]] = node.id

        edges.append(
            Edge(
                source_node_id=doc.id,
                target_node_id=node.id,
                edge_type=EdgeType.ASSERTS,
                meta=base_meta,
            )
        )

    for entity in raw.get("entities", []):
        name = (entity.get("name") or "").strip()
        if not name:
            continue
        entity_type = entity.get("entity_type", "concept")
        node = Node(
            node_type=_map_entity_type(entity_type, schema),
            name=name,
            description=entity.get("description", ""),
            domain=doc.domain,
        )
        nodes.append(node)
        node_name_to_id[name] = node.id

    for rel in raw.get("relationships", []):
        source_name = (rel.get("source_name") or "").strip()
        target_name = (rel.get("target_name") or "").strip()
        rel_type = rel.get("relationship_type", "co_occurs")

        source_id = node_name_to_id.get(source_name)
        target_id = node_name_to_id.get(target_name)

        if source_id and target_id:
            edges.append(
                Edge(
                    source_node_id=source_id,
                    target_node_id=target_id,
                    edge_type=_map_edge_type(rel_type, schema),
                    meta=base_meta,
                )
            )

    return ExtractionResult(
        source_document_id=doc.id,
        nodes=nodes,
        edges=edges,
    )


async def extract_batch(
    documents: list[SourceDocument],
    schema: DomainSchema,
    api_key: str | None = None,
) -> list[ExtractionResult]:
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    client = anthropic.AsyncAnthropic(api_key=api_key)
    results: list[ExtractionResult] = []

    extractable = [d for d in documents if d.abstract or d.full_text]
    skipped = len(documents) - len(extractable)
    if skipped > 0:
        logger.info("Skipping %s documents without text", skipped)

    for i in range(0, len(extractable), MAX_BATCH_SIZE):
        batch = extractable[i : i + MAX_BATCH_SIZE]
        prompt = _build_batch_prompt(schema, batch)

        try:
            response = await client.messages.create(
                model=EXTRACTION_MODEL,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            raw_text = response.content[0].text.strip()
            if raw_text.startswith("["):
                parsed = json.loads(raw_text)
            else:
                parsed = [json.loads(raw_text)]

            for j, doc in enumerate(batch):
                if j < len(parsed) and isinstance(parsed[j], dict):
                    result = _parse_extraction(parsed[j], doc, schema)
                else:
                    result = ExtractionResult(source_document_id=doc.id)
                results.append(result)

        except (json.JSONDecodeError, anthropic.APIError, KeyError, IndexError, TypeError) as e:
            logger.error("Extraction failed for batch %s: %s", i // MAX_BATCH_SIZE, e)
            for doc in batch:
                results.append(ExtractionResult(source_document_id=doc.id))

    total_nodes = sum(len(r.nodes) for r in results)
    total_edges = sum(len(r.edges) for r in results)
    logger.info(
        "Extraction complete: %s docs, %s nodes, %s edges",
        len(results),
        total_nodes,
        total_edges,
    )
    return results
