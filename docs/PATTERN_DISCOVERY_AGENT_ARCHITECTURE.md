# Pattern Discovery Agent

> A domain-agnostic, graph-first pattern discovery engine. Ingests sources, builds knowledge graphs, runs algorithmic pattern detection, and surfaces verified, evidence-backed findings — not summaries.

**Built with:** Python, NetworkX, sentence-transformers, NLI cross-encoders, Anthropic Claude  
**First domain pack:** Research/Technical (OpenAlex, Semantic Scholar, arXiv, GitHub)  
**Runs on:** [RunForge](https://runforge.sh) — deploy agents, not infrastructure  
**License:** MIT

---

## What this is

This is a **pattern discovery engine**, not a research assistant. Given any topic, it:

1. classifies the domain and routes to the right sources,
2. gathers evidence from multiple APIs and data sources,
3. extracts structured assertions, entities, and relationships using LLMs,
4. builds a typed knowledge graph with source provenance on every edge,
5. runs algorithmic pattern detection — community detection, NLI contradiction classification, temporal drift clustering, gap analysis,
6. verifies candidate patterns through a promotion gate with threshold checks,
7. and outputs a pattern report with confidence labels, evidence chains, and explicit blind spots.

The core engine is domain-agnostic. It doesn't know if it's analyzing research papers, sports matchups, political dynamics, or market signals. It sees a graph with typed nodes and edges, and runs the same algorithms regardless of domain. What changes per domain is a **domain pack** — a set of source connectors, extraction prompts, and entity schemas that feed the core.

The first domain pack is **Research/Technical** — the most structured data, the best free APIs, the easiest to evaluate. The architecture is designed so adding new domain packs (sports, politics, markets) requires zero changes to the core engine.

---

## Why this exists

Most "deep research" and "AI analysis" tools follow the same loop: search, read, summarize, write something that sounds insightful. The output rests on whatever the LLM felt like saying. There's no structure, no verification, no way to know if the "insight" is grounded in evidence or hallucinated.

This engine inverts that completely:

```
topic → classify domain → route sources → ingest → extract → build graph → detect patterns → verify → report
```

Patterns are detected by algorithms (Louvain community detection, NLI cross-encoders, HDBSCAN temporal clustering), not invented by the LLM. The LLM does two things: structured extraction (turning raw text into typed nodes and edges) and synthesis (turning verified patterns into readable reports). It never gets to "discover" patterns without evidence underneath.

The verification pipeline is mandatory. A candidate pattern must pass a promotion gate — minimum evidence count, source diversity threshold, confidence scoring, category integrity check — before it appears in the output. If the evidence is too thin, the system says so. If two sources contradict, it shows both sides instead of picking one.

---

## Architecture

### System layers

```
┌──────────────────────────────────────────────────────────────┐
│                      DOMAIN-AGNOSTIC CORE                    │
│                                                              │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────────┐   │
│  │ Graph        │   │ Pattern     │   │ Verifier +      │   │
│  │ Builder      │──>│ Engine      │──>│ Promotion Gate  │   │
│  └─────────────┘   └─────────────┘   └─────────────────┘   │
│        ^                                       │             │
│        │                                       v             │
│  ┌─────────────┐                     ┌─────────────────┐   │
│  │ Universal    │                     │ Report          │   │
│  │ Type System  │                     │ Generator       │   │
│  └─────────────┘                     └─────────────────┘   │
│                                                              │
└──────────────────────────────────────────────────────────────┘
         ^                                       │
         │ (typed nodes + edges)                 │ (artifacts)
         │                                       v
┌──────────────────────────────────────────────────────────────┐
│                       DOMAIN PACK                            │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐   │
│  │ Source        │  │ Extraction   │  │ Domain Entity   │   │
│  │ Connectors   │  │ Prompts      │  │ Schema          │   │
│  └──────────────┘  └──────────────┘  └─────────────────┘   │
│                                                              │
│  Research pack: OpenAlex, Semantic Scholar, arXiv, GitHub     │
│  (Future: Sports pack, Politics pack, Markets pack)          │
└──────────────────────────────────────────────────────────────┘
```

### Full pipeline (9 RunForge safe_steps)

```
Topic input ("AI agent frameworks 2024-2026")
  │
  v
Step 1: Domain classifier — identifies domain, selects pack
  │
  v
Step 2: Source router — picks APIs, sets budgets and depth limits
  │
  ├── OpenAlex ─── papers, citations, semantic search
  ├── Semantic Scholar ─── citation graph, SPECTER2 embeddings
  ├── arXiv + GitHub ─── full text, repositories
  └── Tavily / web ─── technical blogs, docs
  │
  v
Step 3: Corpus ingest — fetch, normalize into SourceDocuments
  │
  v
Step 4: Corpus expand — 1-hop citation/semantic expansion, dedup, tier
  │
  v
Step 5: LLM extraction — entities, assertions, relationships (Haiku batch)
  │                        outputs universal node/edge types
  v
Step 6: Graph build — NetworkX typed graph, entity resolution, metadata
  │
  v
Step 7: Pattern mining
  ├── Bridges ─── Louvain + inter-community semantic similarity
  ├── Contradictions ─── NLI cross-encoder on assertion pairs
  ├── Drift ─── HDBSCAN on temporal embedding windows
  └── Gaps ─── graph queries for unrealized recommendations
  │
  v
Step 8: Verify — promotion gate, confidence scoring, category integrity
  │
  v
Step 9: Report — pattern cards, evidence table, graph viz, coverage
```

Zero commit_step blocks — this agent is read-only, no side effects, no approval gates.

---

## Universal type system

This is the foundation that makes the core domain-agnostic. Every domain pack maps its raw data into these types. The pattern engine, verifier, and report generator only ever see these types.

### Core node types

| Type | What it represents | Research example | Sports example | Politics example |
|------|-------------------|------------------|----------------|-----------------|
| `SourceDocument` | Any ingested source with provenance | Research paper | Game recap, injury report | Bill text, voting record |
| `Assertion` | Any verifiable claim extracted from a source | "Transformers scale better than RNNs" | "Team X 3PT% up 8% last 10 games" | "Senator Y shifted from opposing to supporting" |
| `Actor` | Any entity that takes action | Research lab, author | Player, coach, team | Politician, party, lobbyist |
| `Concept` | Any idea, method, strategy, or category | "multi-head attention", "RLHF" | "zone defense", "small-ball lineup" | "carbon tax", "bipartisan consensus" |
| `Artifact` | Any produced thing | Dataset, repo, benchmark | Season record, draft pick | Legislation, executive order |
| `Metric` | Any measurable quantity with context | Citation count, F1 score | Win rate, defensive rating | Approval rating, vote margin |
| `Event` | Any time-stamped occurrence | Paper publication, benchmark release | Trade, injury, game result | Election, vote, policy announcement |

### Core edge types

| Type | From → To | What it means |
|------|-----------|---------------|
| `asserts` | SourceDocument → Assertion | Source makes this claim |
| `supports` | Assertion → Assertion | One claim supports another |
| `contradicts` | Assertion → Assertion | Claims conflict (NLI-detected) |
| `involves` | Event → Actor | Actor participated in event |
| `produces` | Actor → Artifact | Actor created this thing |
| `evaluates` | Assertion → Artifact | Claim evaluates or benchmarks artifact |
| `associated_with` | Actor → Concept | Actor linked to this concept |
| `extends` | Concept → Concept | One concept builds on another |
| `co_occurs` | Concept → Concept | Appear together frequently |
| `precedes` | Event → Event | Temporal ordering |
| `bridges_to` | any → any | Cross-community link (pattern engine adds) |

### Edge metadata (every edge carries this)

```python
@dataclass
class EdgeMeta:
    source_tier: int            # 1=primary, 2=strong secondary, 3=weak secondary, 4=social
    source_family: str          # "scholarly", "code", "web", "official", "social"
    source_url: str             # provenance URL
    source_id: str              # API-specific ID (OpenAlex ID, S2 corpus ID, etc.)
    extraction_confidence: float  # 0.0-1.0
    extraction_model: str       # "claude-haiku-4-5", "api_metadata", "nli-deberta-v3"
    timestamp: str              # ISO date of the source
    provenance: str             # "abstract", "full_text", "metadata", "api_field"
    domain: str                 # "research", "sports", "politics", etc.
```

### Source tier definitions (universal)

| Tier | Label | Rule | Research | Sports | Politics |
|------|-------|------|----------|--------|----------|
| 1 | Primary evidence | Can anchor a pattern | Peer-reviewed papers, official benchmarks | Official stats, box scores | Voting records, bill text |
| 2 | Strong secondary | Can support, not anchor alone | Credible analysis, replications | Expert analysis, advanced metrics | Nonpartisan analysis (CBO, GAO) |
| 3 | Weak secondary | Discovery only, cannot promote | Vendor blogs, newsletters | Pundit predictions, power rankings | Opinion editorials |
| 4 | Social / anecdotal | Signal layer only | Forum posts, tweets | Fan forums, social media | Social media, partisan commentary |

The rule is the same in every domain: no pattern can be promoted if it rests mainly on Tier 3 or Tier 4 evidence.

---

## Domain pack architecture

A domain pack is a plugin with three components. Adding a new domain means writing these three things. The core never changes.

### 1. Source connectors

Each connector implements:

```python
class SourceConnector:
    def search(self, query: str, limit: int, **kwargs) -> list[SourceDocument]: ...
    def get(self, source_id: str) -> SourceDocument | None: ...
    def expand(self, source_id: str, limit: int) -> list[SourceDocument]: ...
```

Research pack: OpenAlexConnector, SemanticScholarConnector, ArxivConnector, GitHubConnector, TavilyConnector.

Future packs add: ESPNConnector, CongressGovConnector, SECEdgarConnector, etc.

### 2. Extraction schema

Maps domain-specific data to universal types + provides extraction prompts:

```python
class DomainSchema:
    domain: str
    entity_types: dict[str, str]           # domain label -> universal type
    edge_types: dict[str, str]             # domain label -> universal type
    extraction_prompt: str                 # LLM prompt for structured extraction
    tier_rules: dict[str, int]             # source classification rules
    interpretation_templates: dict[str, str]  # pattern explanations per domain
```

### 3. Interpretation templates

The domain pack tells the LLM how to explain patterns in domain-appropriate language:

```python
# Research
"Bridge: Two research communities ({community_a}, {community_b}) discuss similar
mechanisms using different language. {bridge_entity} connects them."

# Sports (future)
"Bridge: Two performance factors ({factor_a}, {factor_b}) correlate but are not
discussed together by analysts. {bridge_entity} connects them."
```

The core engine calls domain_pack.interpret(pattern). It never writes domain-specific language itself.

---

## Pattern mining engine

The engine operates on the universal graph. It doesn't know what domain the nodes came from.

### 1. Bridge detection

Finds entities or concepts weakly connected structurally but strongly related semantically — communities that discuss similar things using different language.

**Algorithm:**
1. Louvain community detection on the graph.
2. Find inter-community edges (structurally weak links).
3. Compute semantic similarity between connected nodes' embeddings. High semantic similarity + low structural connectivity = candidate bridge.
4. Betweenness centrality for bridge candidate nodes.
5. Sonnet validates via domain pack interpretation template.

### 2. Contradiction detection

Finds assertions that conflict — one source says X, another says not-X, or X holds only under specific conditions.

**Algorithm:**
1. Collect all Assertion nodes.
2. Group by topic cluster (Louvain community or embedding similarity).
3. Generate pairs with cosine similarity > 0.6 within each cluster.
4. NLI cross-encoder (cross-encoder/nli-deberta-v3-base) classifies entailment / neutral / contradiction.
5. For pairs classified contradiction with confidence > 0.7, extract conditions.
6. Sonnet evaluates: real contradiction, apparent, or misclassification.

### 3. Drift detection

Detects how a topic evolves over time — new clusters emerge, old approaches decay, phase shifts occur.

**Algorithm:**
1. Bin nodes by time window (configurable).
2. Embed assertion/concept text per window with sentence-transformers.
3. HDBSCAN cluster per window (min_cluster_size=3).
4. Track clusters across windows: birth, growth, death, merge, split.
5. Sonnet labels inflections via domain pack template.

### 4. Gap detection

Finds things frequently recommended but never executed.

**Algorithm:**
1. Find nodes with many incoming "recommends"/"suggests" edges but few/zero "evaluates"/"produces" edges.
2. Score by: recommending source count and tier, recency, adjacent work.
3. Sonnet synthesizes why the gap exists and what barriers remain.

---

## Promotion gate

Every candidate pattern must pass all checks:

| Check | Threshold | Failure action |
|-------|-----------|----------------|
| Minimum evidence | >= 3 distinct evidence items | Withhold or mark "exploratory" |
| Source diversity | >= 2 tiers or >= 3 source URLs | Downgrade confidence |
| Contradiction scan | No self-contradiction in evidence | Flag "unresolved" |
| Category integrity | No merging adjacent categories | Reject or split |
| Confidence threshold | Score >= 0.4 | Mark "low confidence" |
| Blind spots logged | >= 1 blind spot identified | Add default note |

### Confidence model

| Label | Criteria |
|-------|----------|
| **High** | >= 5 evidence items, >= 2 tiers including Tier 1, no contradictions |
| **Medium** | >= 3 evidence items, limited source diversity or minor caveats |
| **Low** | Meets minimum but evidence is thin or diversity poor |
| **Unresolved** | Conflicting evidence too strong to collapse — both sides shown |

System prefers "unresolved" over picking a side. By design.

### Category integrity check

LLM-based. Catches:
- Benchmark wins treated as production wins
- Pilot projects treated as deployed systems
- Vendor claims treated as independent evaluation
- Anecdotes treated as field-level conclusions

---

## Technical decisions and tradeoffs

### Decision 1: NetworkX over graph databases

**Chose:** NetworkX in-memory, serialized via RunForge storage.  
**Rejected:** Neo4j, ArangoDB, JanusGraph.  
**Why:** Agent runs containerized on RunForge. External DB adds cost, latency, failure points. Graphs are 500-5,000 nodes. NetworkX has Louvain, betweenness centrality, shortest paths built in.  
**Tradeoff:** No persistent queryable graph across runs.

### Decision 2: Louvain over Leiden

**Chose:** Louvain (built into NetworkX).  
**Rejected:** Leiden (requires graspologic backend).  
**Why:** Near-identical results under 10K nodes. Zero additional dependencies. One-line swap if needed.  
**Tradeoff:** Slightly less guaranteed community connectivity.

### Decision 3: NLI cross-encoder for contradiction detection

**Chose:** cross-encoder/nli-deberta-v3-base.  
**Rejected:** Cosine similarity alone, LLM-only detection.  
**Why cosine fails:** High cosine ≠ logical agreement. "Scales to 100B" and "fails beyond 10B" have high cosine.  
**Why LLM fails:** 20K possible pairs too slow/expensive. Cross-encoder: 5ms per pair.  
**Tradeoff:** Container needs torch + model weights (~400MB). Adds ~15s cold start.

### Decision 4: sentence-transformers for embeddings

**Chose:** all-MiniLM-L6-v2 local model.  
**Rejected:** OpenAI/Cohere API embeddings, SPECTER2 everywhere.  
**Why:** Local = zero API calls, zero latency, zero cost. SPECTER2 from Semantic Scholar used when available.  
**Tradeoff:** Lower quality on scientific text vs SPECTER2. Mitigated by using S2 embeddings when available.

### Decision 5: HDBSCAN for temporal clustering

**Chose:** HDBSCAN.  
**Rejected:** K-means, BERTopic.  
**Why:** No cluster count needed upfront. Identifies noise points (emerging topics). BERTopic overkill — we only need clusters, LLM labels them.

### Decision 6: Haiku extraction + Sonnet synthesis

**Chose:** Two-tier LLM strategy.  
**Why:** Extraction is 100-300 abstracts needing structured JSON. Haiku at 1/10th Sonnet cost. Synthesis is 5-10 calls needing reasoning. Sonnet's job.  
**Cost:** $0.30-0.60 per 100-document run.

### Decision 7: One unified graph

**Chose:** Single graph, source-tier metadata on edges.  
**Rejected:** Separate linked graphs.  
**Why:** Trust semantics handled by edge metadata for current Tier 1-2 sources. Linked graphs needed only when mixing noisy social sources.  
**Tradeoff:** Known debt if adding social sources later.

### Decision 8: OpenAlex + Semantic Scholar dual primary

**Chose:** Both.  
**Why:** OpenAlex has semantic search + PDFs + topic taxonomy. Semantic Scholar has SPECTER2 + influential citation classifier + TLDRs. Complementary.

### Decision 9: Domain-agnostic core from day one

**Chose:** Universal type system with domain packs.  
**Why:** Cost of abstraction now is near-zero (naming choice). Cost of refactoring later is weeks.

---

## Output format

Each run produces four artifacts via ctx.artifact():

### 1. Pattern report (markdown)

```markdown
### [Bridge/Contradiction/Drift/Gap]: [Title]

**Confidence:** High / Medium / Low / Unresolved
**Domain:** research

**Measured pattern:** [What the algorithm detected — factual]

**Supporting evidence:**
- [Source 1] (Tier X): [What it says]
- [Source 2] (Tier X): [What it says]

**Counterevidence:**
- [Source 3] (Tier X): [What it says differently]

**Interpretation:** [LLM synthesis]

**Blind spots:** [Missing sources, uncovered subtopics]
```

### 2. Evidence table (JSON + CSV)

Every assertion with source, tier, confidence, parent document, relationships.

### 3. Graph visualization (HTML)

D3.js force-directed graph. Nodes colored by community. Bridge edges in amber. Contradiction edges in red. Clickable nodes.

### 4. Coverage report

Source families used, weak subtopics, sparse graph regions, what the agent couldn't find.

---

## RunForge integration

### agent.yaml

```yaml
name: pattern-discovery
entrypoint: agent:run
python_version: "3.11"
browser: false
run_timeout_minutes: 60
max_concurrency: 2
tools: []
```

### State strategy

ctx.state (< 100KB): topic, domain, source_plan, corpus_stats, pattern_stats.

ctx.storage (heavy data): graph.json, documents.json, embeddings.npy, entity_cache, seen_documents.

### Results

```python
ctx.results.set_stats({
    "patterns_found": 7,
    "high_confidence": 3,
    "documents_analyzed": 142,
    "source_coverage": "87%",
    "graph_nodes": 1240,
    "graph_edges": 3890
})
```

### Environment variables

| Variable | Purpose |
|----------|---------|
| ANTHROPIC_API_KEY | Haiku extraction + Sonnet synthesis |
| OPENALEX_API_KEY | OpenAlex API (free key) |
| SEMANTIC_SCHOLAR_API_KEY | Semantic Scholar higher rate limits |
| TAVILY_API_KEY | Optional: web search |
| GITHUB_TOKEN | Optional: higher GitHub rate limits |

### Subscriber inputs

```python
{
    "topic": "AI agent frameworks and orchestration",
    "depth": "standard",        # quick / standard / deep
    "focus": "all",             # bridges / contradictions / drift / gaps / all
    "time_range": "2023-2026",  # optional
    "max_documents": 150,       # default 100
}
```

---

## Project structure

```
pattern-discovery-agent/
├── agent.py                    # RunForge entry point (~30 lines)
├── agent.yaml
├── requirements.txt
├── README.md
│
├── src/
│   ├── __init__.py
│   │
│   ├── core/                   # DOMAIN-AGNOSTIC CORE
│   │   ├── __init__.py
│   │   ├── types.py            # Universal types: SourceDocument, Assertion, Actor,
│   │   │                       # Concept, Artifact, Metric, Event, EdgeMeta
│   │   ├── graph.py            # Graph builder, entity resolution, serialization
│   │   ├── patterns/
│   │   │   ├── __init__.py
│   │   │   ├── bridges.py      # Louvain + semantic similarity
│   │   │   ├── contradictions.py # NLI cross-encoder
│   │   │   ├── drift.py        # HDBSCAN temporal clustering
│   │   │   └── gaps.py         # Graph queries
│   │   ├── verifier.py         # Promotion gate, confidence, category integrity
│   │   └── report.py           # Markdown, evidence table, graph viz, coverage
│   │
│   ├── domain_pack.py          # DomainPack base class + DomainSchema interface
│   │
│   ├── packs/                  # DOMAIN-SPECIFIC PACKS
│   │   ├── __init__.py
│   │   └── research/
│   │       ├── __init__.py
│   │       ├── schema.py       # Entity mapping + extraction prompts
│   │       ├── router.py       # Topic classification
│   │       └── connectors/
│   │           ├── __init__.py
│   │           ├── openalex.py
│   │           ├── semantic_scholar.py
│   │           ├── arxiv.py
│   │           ├── github.py
│   │           └── web_search.py
│   │
│   └── shared/
│       ├── __init__.py
│       ├── corpus.py           # Dedup, expansion, tiering
│       ├── extraction.py       # LLM extraction (domain-pack-aware)
│       └── embeddings.py       # Embedding generation + caching
│
├── tests/
│   ├── core/
│   │   ├── test_types.py
│   │   ├── test_graph.py
│   │   ├── test_bridges.py
│   │   ├── test_contradictions.py
│   │   ├── test_drift.py
│   │   ├── test_gaps.py
│   │   ├── test_verifier.py
│   │   └── test_report.py
│   ├── packs/research/
│   │   ├── test_connectors.py
│   │   ├── test_schema.py
│   │   └── test_router.py
│   ├── shared/
│   │   ├── test_corpus.py
│   │   ├── test_extraction.py
│   │   └── test_embeddings.py
│   └── integration/
│       └── test_full_pipeline.py
│
└── fixtures/
    ├── sample_openalex_response.json
    ├── sample_semantic_scholar_response.json
    ├── sample_corpus.json
    ├── sample_graph.json
    └── sample_assertions.json
```

---

## Implementation phases with test goals

### Phase 1: Universal type system

**Build:** src/core/types.py

```
test_types:
  ✓ SourceDocument serializes to/from JSON without data loss
  ✓ Assertion validates required fields (text, source_tier, source_url)
  ✓ Actor, Concept, Artifact, Metric, Event serialize correctly
  ✓ EdgeMeta validates source_tier is 1-4
  ✓ EdgeMeta validates extraction_confidence is 0.0-1.0
  ✓ All types carry a domain field
  ✓ PatternCandidate holds type, evidence list, confidence, blind spots
  ✓ PromotedPattern extends PatternCandidate with promotion metadata
  ✓ No field references "paper", "claim" — fully domain-agnostic
```

### Phase 2: Domain pack interface + research schema

**Build:** src/domain_pack.py, src/packs/research/schema.py, src/packs/research/router.py

```
test_domain_pack:
  ✓ DomainPack defines required methods: get_connectors, get_schema, interpret
  ✓ ResearchSchema maps "paper" -> SourceDocument, "claim" -> Assertion
  ✓ ResearchSchema extraction prompt produces JSON matching universal types
  ✓ ResearchRouter classifies "transformer architectures" as domain=research
  ✓ Tier rules: peer-reviewed -> Tier 1, arXiv preprint -> Tier 2
  ✓ Interpretation templates produce readable pattern explanations
```

### Phase 3: Source connectors

**Build:** src/packs/research/connectors/

```
test_connectors:
  ✓ OpenAlex returns SourceDocuments with title, abstract, date
  ✓ OpenAlex semantic search finds related papers by abstract
  ✓ OpenAlex citations returns cited-by documents
  ✓ Semantic Scholar batch processes 10+ IDs in one call
  ✓ arXiv returns documents with full abstract
  ✓ GitHub returns repos with stars, language, description
  ✓ All connectors return SourceDocument (universal type)
  ✓ All handle rate limiting with retry + backoff
  ✓ All handle errors without crash (empty + log)
```

### Phase 4: Corpus manager + embeddings

**Build:** src/shared/corpus.py, src/shared/embeddings.py

```
test_corpus:
  ✓ Dedup merges by DOI across sources
  ✓ Dedup merges by title fuzzy match (> 0.9) without DOI
  ✓ Tier assignment follows domain pack tier_rules
  ✓ 1-hop expansion respects budget limit
  ✓ No duplicates after expansion
  ✓ Respects max_documents from input

test_embeddings:
  ✓ MiniLM generates embeddings for text batch
  ✓ SPECTER2 from S2 used when available
  ✓ MiniLM fallback when pre-computed absent
  ✓ Cache avoids re-computing same text
```

### Phase 5: LLM extraction

**Build:** src/shared/extraction.py

```
test_extraction:
  ✓ Uses domain pack extraction prompt (not hardcoded)
  ✓ Returns Assertion nodes with text + conditions
  ✓ Returns Actor, Concept, Artifact nodes
  ✓ Returns typed edges matching universal edge types
  ✓ Batch: 5 texts per LLM call
  ✓ Valid JSON matching type schemas
  ✓ Handles empty text (skip, log, continue)
  ✓ Items carry source_tier + source_url from parent
```

### Phase 6: Graph construction

**Build:** src/core/graph.py

```
test_graph:
  ✓ Contains nodes for all extracted types
  ✓ Edges have correct universal types
  ✓ Every edge has complete EdgeMeta
  ✓ Entity resolution merges "BERT" / "bert" / "Bidirectional Encoder Representations"
  ✓ Does NOT merge genuinely different entities
  ✓ Serializes to JSON and back without data loss
  ✓ < 5MB for 200-document corpus
  ✓ No domain-specific code — fully agnostic
```

### Phase 7: Bridges + contradictions

**Build:** src/core/patterns/bridges.py, contradictions.py

```
test_bridges:
  ✓ Louvain produces >= 2 communities
  ✓ Inter-community edges correctly identified
  ✓ Semantic filter removes low-similarity (< 0.4)
  ✓ Known bridge found on synthetic graph
  ✓ Uses universal types only

test_contradictions:
  ✓ Pairs generated within same cluster only
  ✓ NLI classifies known entailment correctly
  ✓ NLI classifies known contradiction correctly
  ✓ LLM distinguishes real vs apparent contradiction
  ✓ Tier 1 contradictions ranked higher than Tier 3
```

### Phase 8: Drift + gaps

**Build:** src/core/patterns/drift.py, gaps.py

```
test_drift:
  ✓ Temporal binning groups by date correctly
  ✓ HDBSCAN produces clusters per window
  ✓ Detects cluster birth, growth, death
  ✓ Works on generic nodes, not domain-specific

test_gaps:
  ✓ Flags nodes with many "suggests" + zero "evaluates"
  ✓ Weights recommending source tier correctly
  ✓ Excludes recently-closed gaps
```

### Phase 9: Verifier

**Build:** src/core/verifier.py

```
test_verifier:
  ✓ 5 evidence + 2 tiers -> High
  ✓ 3 evidence + 1 tier -> Medium
  ✓ 2 evidence -> withheld
  ✓ Self-contradiction -> Unresolved
  ✓ Category mixing -> flagged
  ✓ Blind spots generated for every promoted pattern
  ✓ Never crashes on edge cases
  ✓ Domain-agnostic — checks counts and tiers, not content
```

### Phase 10: Report + full integration

**Build:** src/core/report.py, agent.py

```
test_report:
  ✓ Markdown has all promoted patterns with required fields
  ✓ Evidence table is valid JSON
  ✓ Graph viz HTML renders D3.js
  ✓ Coverage report identifies weak areas
  ✓ Uses domain pack templates, not hardcoded language

test_integration:
  ✓ End-to-end with python -m agent_runtime dev agent:run
  ✓ All 9 steps complete on test topic
  ✓ Resume from checkpoint works
  ✓ Artifacts produced: markdown, JSON, HTML
  ✓ Run time < 30 min for 100 documents
  ✓ LLM cost < $1.00 for 100 documents
```

---

## Dependencies

```
agent-runtime>=0.1.0
httpx>=0.27.0
networkx>=3.3
sentence-transformers>=3.0.0
scikit-learn>=1.5.0
hdbscan>=0.8.38
numpy>=1.26.0
anthropic>=0.40.0
tenacity>=8.0.0
```

### What we exclude and why

| Package | Why excluded |
|---------|-------------|
| langchain, llamaindex | RAG orchestration we don't use. Massive dep tree. |
| graphrag, lightrag, hipporag | Retrieval over existing corpora. We build from scratch. |
| neo4j, arangodb | External DB. NetworkX handles our sizes. |
| openai | We use Anthropic. |
| playwright, selenium | All sources have APIs. |
| bertopic | Overkill. We need HDBSCAN only. LLM labels topics. |
| spacy, nltk | LLM extraction is more flexible for our use case. |

---

## Expansion hooks (built in, not implemented)

### Adding a new domain pack

1. Create src/packs/{domain}/ with schema.py, router.py, connectors/.
2. Implement DomainPack: get_connectors(), get_schema(), interpret().
3. Map domain entities to universal types.
4. Write extraction prompts + interpretation templates.
5. Core engine works unchanged.

### Future pattern types (designed, not built)

- **Momentum** — signal X precedes outcome Y within timeframe Z. Requires temporal causality + base rate comparison.
- **Divergence** — expert consensus vs market/public pricing. Requires pricing signal source.
- **Leading indicator** — one signal consistently precedes another across the graph.

### Future verification upgrades

- Pattern significance testing (distinguishable from chance?)
- Base rate comparison (4/5 meaningful given base rate?)
- Confidence intervals instead of point estimates
- Bayesian updating across runs

### Future prediction + tracking

- PatternCandidate carries prediction field with verification date.
- ctx.storage persists past predictions and outcomes across runs.
- Calibration tracking: accuracy by confidence level over time.

All additions to the verifier or pattern engine, not changes to the core.

---

## What this project demonstrates

**Systems design** — domain-agnostic core with pluggable domain packs. Clear module contracts.

**Graph theory** — Louvain community detection, betweenness centrality, connected components.

**ML/NLP** — sentence-transformers, NLI cross-encoders, HDBSCAN. Each chosen for documented reasons.

**LLM engineering** — structured extraction, batch processing, two-tier cost strategy.

**Product thinking** — verification, confidence calibration, abstention, category integrity.

**Extensible architecture** — new domains require zero core changes.

**Infrastructure** — RunForge checkpointing, crash recovery, artifact publishing.

---

## License

MIT
