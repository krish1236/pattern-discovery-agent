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

    queries = [topic, f"{topic} survey", f"{topic} benchmark evaluation"]
    if focus in ("drift", "all"):
        queries.append(f"{topic} recent advances")
        queries.append(f"{topic} emerging trends")

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
