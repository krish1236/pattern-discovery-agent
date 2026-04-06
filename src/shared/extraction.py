"""LLM batch extraction into universal graph nodes and edges."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import anthropic

from src.core.types import Edge, EdgeMeta, EdgeType, ExtractionResult, Node, NodeType, SourceDocument
from src.domain_pack import DomainSchema

logger = logging.getLogger(__name__)

# Default: current Haiku (override with ANTHROPIC_EXTRACTION_MODEL)
EXTRACTION_MODEL = os.environ.get("ANTHROPIC_EXTRACTION_MODEL", "claude-haiku-4-5")
# Smaller batches + char budget avoid truncated JSON on long web abstracts
MAX_BATCH_SIZE = 3
MAX_BATCH_TEXT_CHARS = 12_000
EXTRACTION_MAX_TOKENS = int(os.environ.get("ANTHROPIC_EXTRACTION_MAX_TOKENS", "8192"))

_FUTURE_WORK_PATTERN_STRS = [
    r"future work",
    r"future research",
    r"should be explored",
    r"remains to be",
    r"open problem",
    r"further research",
    r"has not been",
    r"yet to be",
    r"underexplored",
    r"under-explored",
    r"needs further",
    r"warrants further",
    r"calls for",
    r"promising direction",
    r"promising avenue",
    r"not yet been",
    r"largely unexplored",
    r"remains unclear",
    r"limited attention",
    r"deserves more",
    r"gap in",
    r"challenge remains",
    r"challenges remain",
    r"not well understood",
    r"poorly understood",
    r"more research is needed",
    r"open challenge",
    r"open question",
    r"unresolved",
    r"worth investigating",
    r"worth exploring",
    r"important direction",
]
_FUTURE_WORK_PHRASES = re.compile(
    "(" + "|".join(_FUTURE_WORK_PATTERN_STRS) + ")",
    re.I,
)
_MIN_ENTITY_NAME_FOR_RECOMMEND = 4


def _message_text(response: Any) -> str:
    """Concatenate all text blocks (Claude may split long replies across multiple blocks)."""
    chunks: list[str] = []
    for block in getattr(response, "content", []) or []:
        if isinstance(block, dict):
            if block.get("type") == "text":
                t = block.get("text")
                if t:
                    chunks.append(str(t))
            continue
        t = getattr(block, "text", None)
        if t:
            chunks.append(t)
    return "".join(chunks).strip()


def _extractable_text_len(doc: SourceDocument) -> int:
    return len(doc.abstract or "") + len(doc.full_text or "")


def chunk_documents_for_extraction(documents: list[SourceDocument]) -> list[list[SourceDocument]]:
    """Split into batches capped by doc count and total text size (prompt + JSON output)."""
    batches: list[list[SourceDocument]] = []
    cur: list[SourceDocument] = []
    used = 0
    for d in documents:
        n = _extractable_text_len(d)
        if cur and (
            len(cur) >= MAX_BATCH_SIZE
            or (used + n > MAX_BATCH_TEXT_CHARS and len(cur) > 0)
        ):
            batches.append(cur)
            cur = []
            used = 0
        cur.append(d)
        used += n
    if cur:
        batches.append(cur)
    return batches


def _parse_extraction_json_payload(raw_text: str) -> list[dict[str, Any]]:
    """Parse JSON array (one object per document) from model output; tolerate fences."""
    t = raw_text.strip()
    # Strip one or more markdown code fences (models often wrap in ```json ... ```).
    for _ in range(5):
        if not t.startswith("```"):
            break
        if "\n" in t:
            t = t.split("\n", 1)[1]
        else:
            t = re.sub(r"^```\w*\s*", "", t, count=1)
        t = t.strip()
        if t.endswith("```"):
            t = t[:-3].strip()
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```\s*$", "", t)
    t = t.strip()
    if not t:
        raise json.JSONDecodeError("Empty model response", t, 0)
    try:
        if t.startswith("["):
            parsed = json.loads(t)
        else:
            parsed = [json.loads(t)]
    except json.JSONDecodeError:
        m = re.search(r"\[[\s\S]*\]", t)
        if m:
            parsed = json.loads(m.group(0))
        else:
            raise
    if not isinstance(parsed, list):
        raise TypeError("expected JSON array")
    return [x for x in parsed if isinstance(x, dict)]


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

    specter: list[float] | None = None
    if doc.precomputed_embedding:
        specter = list(doc.precomputed_embedding)

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
        embedding=specter,
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
        edges.append(
            Edge(
                source_node_id=doc.id,
                target_node_id=node.id,
                edge_type=EdgeType.ASSOCIATED_WITH,
                meta=base_meta,
            )
        )

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

    _add_future_work_recommend_edges(nodes, edges, base_meta)

    return ExtractionResult(
        source_document_id=doc.id,
        nodes=nodes,
        edges=edges,
    )


def _add_future_work_recommend_edges(nodes: list[Node], edges: list[Edge], base_meta: EdgeMeta) -> None:
    """Heuristic RECOMMENDS edges from future-work language in assertions to mentioned entities."""
    seen: set[tuple[str, str]] = set()
    targets = [n for n in nodes if n.node_type in (NodeType.CONCEPT, NodeType.ARTIFACT)]
    for an in nodes:
        if an.node_type != NodeType.ASSERTION:
            continue
        text = (an.properties.get("full_text") or an.description or "").lower()
        if not text or not _FUTURE_WORK_PHRASES.search(text):
            continue
        for tn in targets:
            nm = tn.name.strip()
            if len(nm) < _MIN_ENTITY_NAME_FOR_RECOMMEND:
                continue
            if nm.lower() in text:
                key = (an.id, tn.id)
                if key in seen:
                    continue
                seen.add(key)
                edges.append(
                    Edge(
                        source_node_id=an.id,
                        target_node_id=tn.id,
                        edge_type=EdgeType.RECOMMENDS,
                        meta=base_meta,
                    )
                )


async def extract_batch(
    documents: list[SourceDocument],
    schema: DomainSchema,
    api_key: str | None = None,
) -> list[ExtractionResult]:
    if not documents:
        return []

    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    client = anthropic.AsyncAnthropic(api_key=api_key)
    results: list[ExtractionResult] = []

    extractable = [d for d in documents if d.abstract or d.full_text]
    skipped = len(documents) - len(extractable)
    if skipped > 0:
        logger.info("Skipping %s documents without text", skipped)

    batches = chunk_documents_for_extraction(extractable)
    for batch_idx, batch in enumerate(batches):
        prompt = _build_batch_prompt(schema, batch)

        try:
            response = await client.messages.create(
                model=EXTRACTION_MODEL,
                max_tokens=EXTRACTION_MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )
            raw_text = _message_text(response)
            parsed = _parse_extraction_json_payload(raw_text)

            for j, doc in enumerate(batch):
                if j < len(parsed):
                    result = _parse_extraction(parsed[j], doc, schema)
                else:
                    result = ExtractionResult(source_document_id=doc.id)
                results.append(result)

        except (json.JSONDecodeError, anthropic.APIError, KeyError, IndexError, TypeError) as e:
            logger.error("Extraction failed for batch %s: %s", batch_idx, e)
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
