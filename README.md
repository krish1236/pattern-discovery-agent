# Pattern Discovery Agent

A domain-agnostic, graph-oriented pattern discovery engine designed to run on [RunForge](https://runforge.sh). It ingests sources, builds a typed knowledge graph with provenance, runs algorithmic detectors (bridges, contradictions, temporal drift, gaps), and is intended to surface evidence-backed findings—not generic summaries.

The **research / technical** domain pack is implemented first (OpenAlex, Semantic Scholar, arXiv, optional GitHub and Tavily). The **core type system and pipeline contracts** are domain-neutral so additional packs can be added without changing the graph and verification layers.

## Documentation

| Document | Purpose |
|----------|---------|
| [docs/PATTERN_DISCOVERY_AGENT_ARCHITECTURE.md](docs/PATTERN_DISCOVERY_AGENT_ARCHITECTURE.md) | End-to-end architecture, type system, pattern algorithms, RunForge integration |
| [docs/PATTERN_DISCOVERY_IMPLEMENTATION_PART1.md](docs/PATTERN_DISCOVERY_IMPLEMENTATION_PART1.md) | Phases 1–5: types, domain pack, connectors, corpus, extraction |
| [docs/PATTERN_DISCOVERY_IMPLEMENTATION_PART2.md](docs/PATTERN_DISCOVERY_IMPLEMENTATION_PART2.md) | Phases 6–10: graph, miners, verifier, report, full `agent.py` |

## Current implementation status

**Implemented**

- Universal types (`src/core/types.py`): nodes, edges, `EdgeMeta`, `SourceDocument`, extraction and pattern artifacts
- Domain pack interface (`src/domain_pack.py`) and **research pack** (`src/packs/research/`)
- Source connectors: OpenAlex, Semantic Scholar, arXiv, GitHub (token), Tavily web search (API key)
- Corpus helpers: deduplication, expansion hook, tier assignment, stats (`src/shared/corpus.py`)
- Embeddings utilities: MiniLM via sentence-transformers (`src/shared/embeddings.py`)
- LLM batch extraction into universal nodes/edges, with assertion provenance for downstream evidence (`src/shared/extraction.py`)
- Unit tests for types, research schema/router/pack, connector parsing, corpus, extraction
- RunForge entrypoint (`agent.py`): topic classification and source routing steps wired; full ingest → graph → mine → verify → artifacts to be connected per Part 2

**Not yet wired in this repository**

- `KnowledgeGraph` (NetworkX), four pattern detectors, promotion gate, report/HTML graph artifacts, and end-to-end `safe_step` sequence as specified in Part 2 of the implementation docs

## Repository layout

```
pattern-discovery/
├── agent.py                 # RunForge entrypoint
├── agent.yaml               # Agent manifest for RunForge
├── pyproject.toml
├── requirements.txt
├── docs/                    # Architecture and phased implementation specs
├── src/
│   ├── core/                # Types (graph/patterns/verifier/report added in Part 2)
│   ├── domain_pack.py
│   ├── packs/research/      # Schema, router, connectors, ResearchPack
│   └── shared/              # corpus, embeddings, extraction
└── tests/
```

## Requirements

- Python **3.11+**
- [agent-runtime](https://pypi.org/project/agent-runtime/) (declared as `>=0.0.1` on PyPI). If you rely on APIs newer than the published package, install a local checkout first, for example:

  ```bash
  pip install -e ../agent-runtime
  ```

Heavy optional runtime dependencies (pull in PyTorch via sentence-transformers): see `pyproject.toml`.

## Environment variables

| Variable | Required | Role |
|----------|----------|------|
| `ANTHROPIC_API_KEY` | Yes, for extraction | Claude Haiku (extraction) |
| `OPENALEX_API_KEY` | No | OpenAlex (if using authenticated usage) |
| `SEMANTIC_SCHOLAR_API_KEY` | No | Higher S2 rate limits |
| `GITHUB_TOKEN` | No | GitHub search quota |
| `TAVILY_API_KEY` | No | Web search connector |

## Local development

Create a virtual environment and install the package with dev tools:

```bash
cd pattern-discovery
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Run tests (fast path, no live APIs or slow model-heavy cases):

```bash
PYTHONPATH=. pytest tests/core/test_types.py tests/packs/research/ tests/shared/ -v
```

Optional markers (when you add them to tests): `-m "not slow and not live and not integration"`.

## Running the agent locally

```bash
python -m agent_runtime dev agent:run
```

Ensure `agent.yaml` `entrypoint` matches your module (`agent:run`). Subscriber-style inputs are merged from both the trigger payload and `ctx.inputs` (dashboard experience layer), for example:

```json
{
  "topic": "AI agent frameworks and orchestration",
  "depth": "standard",
  "focus": "all",
  "time_range": "2023-2026",
  "max_documents": 100
}
```

## License

MIT (see architecture doc; add a `LICENSE` file when you publish).

## Contributing

Follow the phased plan in `docs/PATTERN_DISCOVERY_IMPLEMENTATION_PART2.md` for graph construction, pattern mining, verifier, reports, and full RunForge `ctx.storage` / `ctx.artifact` integration.
