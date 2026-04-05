"""Research domain pack schema and prompts."""

from src.domain_pack import DomainSchema

RESEARCH_EXTRACTION_PROMPT = """You are extracting structured knowledge from a research document.

Given the following text (title + abstract), extract:

1. ASSERTIONS: Verifiable claims the document makes.
   Each assertion needs: text, conditions (if any), polarity (positive/negative/neutral).

2. ENTITIES: Named things mentioned.
   Each entity needs: name, entity_type (one of: actor, concept, artifact, metric).
   - actor: people, labs, organizations, companies
   - concept: methods, techniques, ideas, approaches, algorithms
   - artifact: datasets, benchmarks, tools, repositories, models
   - metric: measurable quantities (accuracy, F1, latency, cost)

3. RELATIONSHIPS: How entities and assertions connect.
   Each relationship needs: source_name, target_name, relationship_type.
   Allowed relationship types: supports, contradicts, extends, evaluates,
   produces, associated_with, co_occurs, recommends.

Respond ONLY with valid JSON in this exact format:
{
  "assertions": [
    {"text": "...", "conditions": "...", "polarity": "positive"}
  ],
  "entities": [
    {"name": "...", "entity_type": "concept", "description": "..."}
  ],
  "relationships": [
    {"source_name": "...", "target_name": "...", "relationship_type": "supports"}
  ]
}

Do NOT include any text outside the JSON. Do NOT wrap in markdown code blocks.
If the text has no extractable content, return {"assertions": [], "entities": [], "relationships": []}.

Text to extract from:
---
Title: {title}
Abstract: {abstract}
---"""


RESEARCH_INTERPRETATION_TEMPLATES = {
    "bridge": (
        "Two research communities ({community_a} and {community_b}) are working on "
        "related problems using different terminology and rarely citing each other. "
        "The connecting thread is {bridge_entity}. "
        "Explain why this bridge matters and what each community could learn from the other."
    ),
    "contradiction": (
        'Source A claims: "{assertion_a}" (from {source_a}). '
        'Source B claims: "{assertion_b}" (from {source_b}). '
        "These assertions appear to contradict. Determine: "
        "1) Is this a real contradiction or an apparent one (different conditions, definitions, or populations)? "
        "2) What conditions make each claim true? "
        "3) What additional evidence would resolve this?"
    ),
    "drift": (
        "This research area has evolved across these time windows: {time_windows}. "
        "The following cluster transitions were detected: {transitions}. "
        "Describe the phase shift: what was the field doing before, what changed, and what is emerging now."
    ),
    "gap": (
        "The following concept/method/experiment is frequently recommended across {recommender_count} sources "
        "but has {execution_count} actual results in the literature: {gap_description}. "
        "Explain why this gap likely exists and what barriers prevent execution."
    ),
}

RESEARCH_TIER_RULES = {
    "peer_reviewed": 1,
    "official_benchmark": 1,
    "official_documentation": 1,
    "first_party_report": 1,
    "arxiv_preprint": 2,
    "credible_analysis": 2,
    "implementation_writeup": 2,
    "benchmark_replication": 2,
    "vendor_blog": 3,
    "newsletter": 3,
    "general_article": 3,
    "forum_post": 4,
    "social_media": 4,
    "unverified_commentary": 4,
}

RESEARCH_ENTITY_TYPES = {
    "paper": "source_document",
    "claim": "assertion",
    "author": "actor",
    "lab": "actor",
    "organization": "actor",
    "method": "concept",
    "technique": "concept",
    "algorithm": "concept",
    "dataset": "artifact",
    "benchmark": "artifact",
    "model": "artifact",
    "repository": "artifact",
    "tool": "artifact",
    "citation_count": "metric",
    "accuracy": "metric",
    "f1_score": "metric",
    "publication": "event",
    "benchmark_release": "event",
}

RESEARCH_EDGE_TYPES = {
    "cites": "asserts",
    "claims": "asserts",
    "authored_by": "involves",
    "evaluated_on": "evaluates",
    "implemented_in": "produces",
    "builds_on": "extends",
    "related_to": "co_occurs",
    "suggests_future_work": "recommends",
}


def get_research_schema() -> DomainSchema:
    return DomainSchema(
        domain="research",
        entity_types=RESEARCH_ENTITY_TYPES,
        edge_types=RESEARCH_EDGE_TYPES,
        tier_rules=RESEARCH_TIER_RULES,
        extraction_prompt=RESEARCH_EXTRACTION_PROMPT,
        interpretation_templates=RESEARCH_INTERPRETATION_TEMPLATES,
    )
