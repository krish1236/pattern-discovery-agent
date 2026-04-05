"""Research domain router and source planning."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SourcePlan:
    connectors: list[str]
    queries: list[str]
    max_documents: int = 100
    expansion_budget: int = 50
    per_connector_limit: int = 30
    time_filter: str | None = None


RESEARCH_KEYWORDS = [
    "research",
    "paper",
    "study",
    "algorithm",
    "framework",
    "model",
    "benchmark",
    "dataset",
    "architecture",
    "neural",
    "transformer",
    "machine learning",
    "deep learning",
    "NLP",
    "computer vision",
    "reinforcement learning",
    "optimization",
    "distributed systems",
    "database",
    "compiler",
    "operating system",
    "network protocol",
    "security",
    "cryptography",
    "software engineering",
    "API",
    "microservice",
    "kubernetes",
    "cloud",
    "serverless",
    "edge computing",
    "LLM",
    "GPT",
    "BERT",
    "diffusion",
    "generative",
    "agent",
    "retrieval",
    "RAG",
    "knowledge graph",
    "embedding",
    "fine-tuning",
    "evaluation",
    "scaling",
    "inference",
    "training",
    "pretraining",
]


def is_research_topic(topic: str) -> bool:
    topic_lower = topic.lower()
    return any(kw in topic_lower for kw in RESEARCH_KEYWORDS)


def _generate_queries(topic: str, max_queries: int = 8) -> list[str]:
    """Split a natural-language topic into diverse sub-queries for scholarly APIs."""
    queries = [topic.strip()]
    words = topic.lower().replace(",", " ").split()
    filler = {
        "and",
        "or",
        "the",
        "of",
        "in",
        "for",
        "with",
        "vs",
        "versus",
        "a",
        "an",
        "to",
        "on",
        "debate",
    }
    key_phrases = [w for w in words if w not in filler and len(w) > 2]
    if len(key_phrases) >= 2:
        for i in range(0, len(key_phrases) - 1):
            queries.append(f"{key_phrases[i]} {key_phrases[i + 1]}")
    queries.append(f"{topic} survey")
    queries.append(f"{topic} review")
    seen: set[str] = set()
    unique: list[str] = []
    for q in queries:
        k = q.lower().strip()
        if k and k not in seen:
            seen.add(k)
            unique.append(q.strip())
    return unique[:max_queries]


def build_source_plan(
    topic: str,
    depth: str = "standard",
    focus: str = "all",
    time_range: str | None = None,
    max_documents: int = 100,
) -> SourcePlan:
    depth_limits = {"quick": 50, "standard": 100, "deep": 200}
    max_docs = depth_limits.get(depth, max_documents)

    connectors = ["openalex", "semantic_scholar", "arxiv"]
    code_keywords = ["framework", "library", "tool", "sdk", "api", "implementation"]
    if any(kw in topic.lower() for kw in code_keywords):
        connectors.append("github")
    connectors.append("web_search")

    queries = _generate_queries(topic)
    if f"{topic} benchmark evaluation".lower() not in {q.lower() for q in queries}:
        queries.append(f"{topic} benchmark evaluation")
    if focus in ("drift", "all"):
        for extra in (f"{topic} recent advances", f"{topic} emerging trends"):
            if extra.lower() not in {q.lower() for q in queries}:
                queries.append(extra)
    queries = queries[:10]

    per_connector = max(10, max_docs // max(len(connectors), 1))
    expansion_budget = max(20, max_docs // 3)

    return SourcePlan(
        connectors=connectors,
        queries=queries,
        max_documents=max_docs,
        expansion_budget=expansion_budget,
        per_connector_limit=per_connector,
        time_filter=time_range,
    )
