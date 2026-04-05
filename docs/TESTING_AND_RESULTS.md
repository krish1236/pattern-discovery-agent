# Testing layout and results

## Where test results appear

| What | Where |
|------|--------|
| **Pytest exit status and per-test output** | Your terminal (stdout/stderr) when you run `pytest`. There is no checked-in “report file” unless you add one (e.g. `--junitxml=...`). |
| **Pytest cache** | `.pytest_cache/` (local only; listed in `.gitignore` patterns via normal cache ignores). |
| **Evaluation / quality pipeline artifacts** | `eval_output/<timestamp>/` after `python scripts/run_evaluation.py`. Used as input for **`tests/quality/`**. That directory is **gitignored** (`eval_output/`). |
| **CI** | If you add GitHub Actions (or similar), results would show in the workflow log and any uploaded artifacts you configure. |

To capture a log locally:

```bash
pytest tests/ -m "not slow and not live and not quality" -v --tb=short 2>&1 | tee pytest-main.log
pytest tests/quality/ -m quality -v --tb=short 2>&1 | tee pytest-quality.log
```

---

## Test inventory (by area)

| Path | Role |
|------|------|
| `tests/core/` | Types, graph, patterns (bridges, gaps, drift, contradictions), verifier, report, pack registry |
| `tests/shared/` | Corpus, embeddings, extraction, checkpoint |
| `tests/packs/research/` | Schema, connectors (fixtures), interpret formatting, optional **live** HTTP (`test_connectors_live.py`) |
| `tests/integration/` | Graph round-trip + pack smoke; **mocked `agent.run`** full-step run and resume checkpoint |
| `tests/quality/` | Assertions on **real** `eval_output/*` (corpus, extraction summary, graph stats, reports). Marked **`quality`**; requires a prior `scripts/run_evaluation.py` run. |

Pytest markers (see `pyproject.toml`):

- **`slow`** — heavy models (e.g. NLI in contradiction tests)
- **`live`** — real API calls
- **`integration`** — integration scenarios
- **`quality`** — needs `eval_output/` (or `EVAL_DIR`)

---

## What changed recently in tests (summary)

- **`tests/integration/test_full_pipeline.py`**: Stub `agent_runtime` when missing; **`MockCtx`** + patched connectors / `extract_batch` / `embed_texts`; tests full **`agent.run`** and **resume-from-checkpoint** without live APIs.
- **`tests/core/test_contradictions.py`**: Async tests for **`refine_contradiction_candidates`** (mocked Anthropic): drops “apparent”, keeps “real”.
- **`tests/shared/test_embeddings.py`**: Cache reset in autouse fixture; test that **second identical encode** does not call the model twice.
- **`tests/shared/test_corpus.py`**: **`expand_corpus([], [])`** returns `[]` immediately.
- **`tests/shared/test_extraction.py`**: **`precomputed_embedding`** copied to source-document node; **`_parse_extraction_json_payload`** markdown-fence parsing.
- **`tests/quality/test_output_quality.py`**: Moved from `docs/`; module marker **`quality`**; **`_tier_count`** for JSON tier keys; skips when corpus is single-family / no tier-1 / **`focus != all`** for pattern-count tests.

---

## Latest result snapshots (local runs)

These numbers are from a clean run on the development machine used while updating the repo. **Re-run the commands below** on yours to refresh.

### Main suite (unit + integration, no slow/live/quality)

```bash
cd /pattern-discovery/repo/root
.venv/bin/pytest tests/ -m "not slow and not live and not quality" -q
```

**Snapshot:** `110 passed`, `37 deselected` (markers filter out slow/live/quality).

### Quality suite (needs `eval_output/` from `scripts/run_evaluation.py`)

```bash
.venv/bin/pytest tests/quality/ -m quality -v
```

**Snapshot (with a `focus=gaps` eval present):** `17 passed`, `14 skipped`, `1 xfailed`.

- **Skipped** — conditions documented in tests (e.g. no tier-1 docs, single source family, `focus != all`, no pattern cards in report).
- **Xfail** — `test_evidence_table_not_empty` when there are zero patterns (empty evidence table is expected in that case).

### Slow / live (not run in the snapshot above)

```bash
.venv/bin/pytest tests/ -m slow -q          # e.g. NLI model
RUN_LIVE_CONNECTOR_TESTS=1 .venv/bin/pytest tests/packs/research/test_connectors_live.py -m live -q
```

---

## Related scripts

- **`scripts/run_evaluation.py`** — end-to-end pipeline writing **`eval_output/<timestamp>/`** (loads `.env` from repo root when variables are unset).

---

## Keeping this document accurate

After meaningful test or marker changes, re-run the two pytest commands in “Latest result snapshots” and update the pass/skip/xfail/deselect counts in this file.
