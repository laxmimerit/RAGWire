# RAGWire — Production Readiness Tracker

> Audit date: **2026-07-25** · Version audited: **1.3.2** · Branch: `main`
>
> Guiding constraint: **every change must keep the 3-line setup story intact.**
> `pip install ragwire` → `config.yaml` → `RAGWire("config.yaml")`. Nothing below
> may add a required step for the beginner path. Production features are opt-in.

**Status legend:** ☐ open · ◐ in progress · ☑ done · ✗ won't fix

---

## 1. Confirmed bugs

Ordered by blast radius. Each entry has a concrete failure scenario so it can be
turned into a regression test.

### B1 — `compare_hashes()` raises `AttributeError` on every call ☐
**File:** [ragwire/processing/hashing.py:116](ragwire/processing/hashing.py#L116) · **Severity:** high

```python
return hashlib.compare_digest(...)   # hashlib has no compare_digest
```

`compare_digest` lives in `hmac`, not `hashlib`. Verified:

```
>>> compare_hashes('a'*64, 'a'*64)
AttributeError: module 'hashlib' has no attribute 'compare_digest'
```

The function is exported-adjacent public API and is 100% dead-on-arrival. Nothing
in the pipeline calls it, which is exactly why it was never caught — it is a
direct signal that the module has no unit test.

**Fix:** `import hmac` → `hmac.compare_digest(...)`. Add a test.

---

### B2 — Global `warnings.filterwarnings("ignore")` on import ✗
**File:** [ragwire/utils/logging.py:14](ragwire/utils/logging.py#L14) · **Won't fix — intentional**

Importing `ragwire` mutates the host process's global warning filters. Kept
deliberately to suppress LangChain/dependency noise for the beginner path.
Logged here so it is not re-reported as a defect.

---

### B3 — Failed ingestion leaves a permanently half-ingested document ☐
**File:** [ragwire/core/pipeline.py:345](ragwire/core/pipeline.py#L345) · **Severity:** high

```python
self.vectorstore.add_documents(chunks)   # may fail partway through
...
except Exception as e:
    stats["failed"] += 1
```

`add_documents` batches internally. If it fails after the first batch is committed
(network blip, Qdrant restart, rate limit), some chunks carrying `file_hash=H` are
now in the collection. On the next run, `file_hash_exists(H)` returns `True` and
the file is **skipped as "already ingested."**

**Scenario:** a 400-chunk 10-K, connection drops at chunk 150. Run reports
`failed: 1`. User re-runs → `skipped: 1`. The collection permanently contains 150
of 400 chunks and every subsequent retrieval silently misses the back half of the
document. There is no command to recover from this state.

**Fix:** make ingestion atomic per file. Write chunks, then write a completion
marker (e.g. a `ingest_complete: true` payload on a sentinel point, or upsert the
`file_hash` record *last*). `file_hash_exists` must check the marker, not any
chunk. Add `rag.reingest(path, force=True)` and a `rag.repair()` that finds file
hashes whose stored chunk count ≠ `total_chunks`.

---

### B4 — LLM metadata failure is swallowed and becomes permanent ☐
**File:** [ragwire/core/pipeline.py:441-446](ragwire/core/pipeline.py#L441) · **Severity:** high

```python
try:
    llm_metadata = self.extract_metadata(text)
except Exception as e:
    logger.warning(f"LLM metadata extraction failed for {file_name}: {e}")
    # llm_metadata stays {} — ingestion proceeds
```

A transient LLM timeout means the document is ingested with **zero semantic
metadata**. It still counts as `processed`, so the user sees a green run. Because
of file-hash dedup (B3), re-running ingestion **cannot fix it** — the file is
skipped forever.

**Scenario:** Ollama is mid-model-swap during a 500-file batch. 40 files ingest
with no `company_name`/`fiscal_year`. Every metadata-filtered query silently
excludes them. The only recovery is wiping the collection.

**Fix:** retry with backoff (2–3 attempts), then either (a) fail the file so it is
retried on the next run, or (b) ingest it but tag `metadata_status: "failed"` so
it is discoverable and repairable. Surface a `metadata_failed` counter in
`IngestStats`.

---

### B5 — Files producing zero chunks vanish from the stats ☐
**File:** [ragwire/core/pipeline.py:346-350](ragwire/core/pipeline.py#L346) · **Severity:** medium

```python
if chunks:
    self.vectorstore.add_documents(chunks)
    stats["chunks_created"] += len(chunks)
    stats["processed"] += 1
# no else branch
```

If `split_text` returns `[]` — scanned PDF with no text layer, empty `.txt`,
password-protected DOCX that MarkItDown returns empty for — the file is counted in
`total` but in **none** of `processed` / `skipped` / `failed`.

**Scenario:** `ingest_directory("data/")` on 100 files where 12 are image-only
scans returns `{total: 100, processed: 88, skipped: 0, failed: 0}`. The numbers
do not add up and the user has no way to learn which 12 files are missing. This is
the most common real-world RAG failure (scanned PDFs) and the framework hides it.

**Fix:** add an `else` branch → `stats["failed"] += 1` with error
`"no extractable text (possibly a scanned/image-only document)"`. Assert
`processed + skipped + failed == total` in tests.

---

### B6 — Custom integer metadata fields get the wrong Qdrant index ☐
**File:** [ragwire/vectorstores/qdrant_store.py:283](ragwire/vectorstores/qdrant_store.py#L283) · **Severity:** high

```python
_INTEGER_FIELDS = {"chunk_index", "total_chunks", "fiscal_year"}
```

The integer-vs-keyword decision is a **hardcoded allowlist of the built-in
financial schema**. Custom schemas via `metadata.config_file` are a headline
feature, and any custom field declared `type: integer` (e.g. `publication_year`,
`revision`, `page_count`) falls through to `PayloadSchemaType.KEYWORD`.

A keyword index does not index integer payload values. The subsequent
`create_payload_index` call fails or produces a useless index — and the failure is
swallowed by the bare `except` on line 311 (see B7). Downstream, `client.facet()`
returns nothing, so `get_field_values()` returns `[]` for that field, so
`_extract_filters_from_query` shows the LLM an empty value list and any filter it
produces matches zero points.

**Scenario:** `examples/metadata.yaml`-style config with
`- name: publication_year, type: integer`. User asks *"papers from 2023"*.
`extract_filters` returns `{"publication_year": 2023}`, `retrieve()` builds a
valid filter, Qdrant returns **0 documents**, and the agent answers "no relevant
documents found" for a corpus that is full of them.

**Fix:** thread the declared field types from `MetadataExtractor.fields` /
the YAML into `create_payload_indexes`. Fall back to inferring the schema type
from a sampled payload value rather than a hardcoded name set.

---

### B7 — Bare `except: pass` hides index-creation failures ☐
**File:** [ragwire/vectorstores/qdrant_store.py:311-312](ragwire/vectorstores/qdrant_store.py#L311) · **Severity:** medium

```python
except Exception:
    pass  # Already exists — safe to ignore
```

The comment assumes the only possible exception is "already exists." It also
swallows auth failures, connection resets, quota errors, and the wrong-schema-type
error from B6. Every payload index can silently fail to exist and the pipeline
reports success.

**Fix:** catch narrowly — inspect the Qdrant error for the already-exists case,
`logger.debug` that, and `logger.warning` (or re-raise) everything else.

---

### B8 — Metadata field discovery samples exactly one point ☐
**File:** [ragwire/vectorstores/qdrant_store.py:269-279](ragwire/vectorstores/qdrant_store.py#L269) · **Severity:** medium

```python
results, _ = self.client.scroll(collection_name=..., limit=1, ...)
metadata = payload.get("metadata", {})
return list(metadata.keys())
```

Field discovery reads **one arbitrary point** and treats its keys as the schema of
the entire collection. Two things break:

1. Fields that were `None` for that one document are absent from its payload, so
   they never get a payload index → `get_field_values` fails for them.
2. Mixed-schema collections (built-in financial docs ingested first, then a custom
   schema) only ever expose whichever schema the sampled point belongs to.

**Scenario:** collection where document #1 is an 8-K (`fiscal_quarter: None`).
`discover_metadata_fields()` omits `fiscal_quarter` entirely, so no index is
created, so filtering on quarter returns nothing across the whole collection.

**Fix:** sample N points (e.g. 100) and union the keys, or — better — derive the
field list from the configured schema (`self._filter_fields`) plus the known
system fields, which is authoritative and needs no network call.

---

### B9 — Caller-supplied filters are not normalized; LLM-extracted ones are ☐
**File:** [ragwire/core/pipeline.py:648-651](ragwire/core/pipeline.py#L648) · **Severity:** medium

`_extract_filters_from_query` lowercases every extracted string value
([pipeline.py:609-613](ragwire/core/pipeline.py#L609)), and `MetadataExtractor.extract`
lowercases everything on write. But `retrieve(query, filters=...)` passes the
caller's dict straight to `_build_qdrant_filter` with **no normalization**.

**Scenario:** the documented agent pattern in
[examples/rag_agent.py:58](examples/rag_agent.py#L58) hands the LLM's tool-call
arguments directly to `rag.retrieve`. The agent writes
`filters={"company_name": "Apple Inc."}`. Stored value is `"apple inc."`.
Qdrant `MatchValue` is exact and case-sensitive → **0 results**. The agent
concludes the corpus has nothing about Apple.

**Fix:** run caller-supplied filters through the same normalization as the
extracted path — one shared `_normalize_filters()` helper used by both.

---

### B10 — Dependency floors permit versions the code cannot run on ☐
**File:** [pyproject.toml:43-46](pyproject.toml#L43) · **Severity:** high

```toml
"langchain>=0.1.0",
"langchain-core>=0.1.0",
"langchain-community>=0.0.0",
"langchain-text-splitters>=0.0.1",
"langchain-qdrant>=0.1.0",
"markitdown[pdf]>=0.0.1",
```

These floors are decorative. The code requires much newer APIs:

| Code | Requires |
|---|---|
| `response.text` as a **property** ([pipeline.py:604](ragwire/core/pipeline.py#L604)) | `langchain-core` 1.x — it was a *method* in 0.3.x and absent in 0.1.x |
| `llm.with_structured_output()` ([extractor.py:108](ragwire/metadata/extractor.py#L108)) | `langchain-core` ≫ 0.1.0 |
| `RetrievalMode.HYBRID`, `FastEmbedSparse` ([qdrant_store.py:109](ragwire/vectorstores/qdrant_store.py#L109)) | `langchain-qdrant` ≫ 0.1.0 |

Locally installed `langchain-core` is 1.4.8, which is why it works on the author's
machine. A fresh resolve in a constrained environment (an existing project pinning
older LangChain) will install a permitted-but-broken combination and fail with
`AttributeError: 'function' object has no attribute 'strip'` deep inside filter
extraction. Optional extras are worse — every one is `>=0.0.0`.

**Fix:** set real floors from what is actually tested (`langchain-core>=1.0`,
`langchain-qdrant>=0.2`, etc.), add upper bounds on majors, and commit a
`requirements-lock.txt` used by CI.

---

### B11 — Shipped `config.yaml` wipes the collection on every startup ☐
**File:** [config.yaml:52](config.yaml#L52) · **Severity:** high

```yaml
force_recreate: true  # Set true ONLY during testing to wipe and rebuild the collection
```

`config.yaml` is **tracked in git** and is the file every example loads
(`RAGWire("config.yaml")` in [examples/rag_agent.py:28](examples/rag_agent.py#L28)).
`config.example.yaml:82` and all 14 docs pages correctly say `false`. The one file
users actually run says `true`.

**Scenario:** user ingests 2,000 documents, restarts their service the next day,
and `_initialize_vectorstore` calls `delete_collection()` before they issue a
single query. Total silent data loss on a plain restart, with no prompt and only
an INFO log line.

**Fix:** flip to `false` in the tracked `config.yaml`. Consider removing tracked
`config.yaml` entirely and having docs say "copy `config.example.yaml`", so there
is exactly one canonical template. Additionally: `force_recreate` should require a
second confirmation (env var, or `RAGWire(..., allow_destructive=True)`) before it
deletes a non-empty collection.

---

### B12 — `log_level: DEBUG` never reaches the console ☐
**File:** [ragwire/utils/logging.py:56](ragwire/utils/logging.py#L56), [:135](ragwire/utils/logging.py#L135) · **Severity:** low

```python
console_handler.setLevel(logging.INFO)   # hardcoded, ignores log_level
```

Both `setup_logging` and `setup_colored_logging` set the logger to the requested
level but pin the console handler to INFO. Setting `logging.level: "DEBUG"` in
config produces no additional console output — DEBUG records are filtered at the
handler. Every `logger.debug` in the codebase (extracted metadata, payload index
creation, chunk details) is unreachable without a log file.

**Scenario:** user debugging why metadata extraction returns nulls sets
`level: DEBUG`, sees no change, and has no way to inspect what the LLM returned.

**Fix:** `console_handler.setLevel(getattr(logging, log_level.upper()))`.

---

### B13 — No embedding-dimension check against an existing collection ☐
**File:** [ragwire/core/pipeline.py:261-267](ragwire/core/pipeline.py#L261) · **Severity:** medium

When the collection already exists, RAGWire attaches to it without comparing the
stored `vector_size` to the dimension of the currently configured embedding model.

**Scenario:** user switches `embeddings.model` from `nomic-embed-text` (768) to
`qwen3-embedding:0.6b` (1024). Init succeeds and logs "Using existing collection."
The first `add_documents` fails with a raw Qdrant dimension error, or — worse, for
some client paths — queries return meaningless neighbours. The README's fix
("set `force_recreate: true` once") is only discoverable after the failure.

**Fix:** in `_initialize_vectorstore`, compare `len(embedding.embed_query("test"))`
against `get_collection_info().config.params.vectors.size` and raise a clear,
actionable error naming both dimensions and the exact remediation.

---

### B14 — `DocumentMetadata` is declared, exported, and never used ☐
**File:** [ragwire/metadata/schema.py:13](ragwire/metadata/schema.py#L13) · **Severity:** low

`_process_document` builds a **raw dict** ([pipeline.py:453](ragwire/core/pipeline.py#L453))
and never validates it against `DocumentMetadata`. The model is exported in
`__all__` and tested in `tests/test_imports.py`, giving a false impression that
ingestion output is schema-validated. It also *contradicts* the real schema:
`DocumentMetadata.fiscal_year: Optional[List[int]]` vs
`FinancialMetadata.fiscal_year: Optional[int]`
([extractor.py:47](ragwire/metadata/extractor.py#L47)) — and `_INTEGER_FIELDS`
indexes it as a scalar integer (B6). Three sources of truth, two of them wrong.

**Fix:** either validate chunk metadata through `DocumentMetadata` in
`_process_document` (and reconcile `fiscal_year`), or delete the model. Do not
ship an unenforced schema.

---

### B15 — `extract()` truncates at 4,000 chars while docs promise 10,000 ☐
**File:** [ragwire/metadata/extractor.py:140](ragwire/metadata/extractor.py#L140) · **Severity:** medium

```python
result = chain.invoke({"content": text[:4000]})
```

The docstring one line above says *"first 10,000 chars used"*
([extractor.py:116](ragwire/metadata/extractor.py#L116)), and
[pipeline.py:438](ragwire/core/pipeline.py#L438) comments *"capped at 10k chars in
extract()"* while explicitly justifying passing full text instead of chunk 0.

4,000 characters is roughly one page of a 10-K. `fiscal_year` ("for the fiscal
year ended…") and `fiscal_quarter` commonly appear on the cover page, but
`doc_type` mapping evidence and any custom field defined further into the document
will be outside the window. Silent metadata loss on exactly the long documents
this framework targets.

**Fix:** make the cap a constructor/config parameter (`extraction_char_limit`,
default 10000), and align the docstrings. Better: derive the cap from the model's
context window rather than a magic number.

---

## 2. Production-grade gaps (not bugs — missing capability)

| # | Gap | Why it blocks production | Keeps setup simple? |
|---|---|---|---|
| G1 | **No CI test job.** `.github/workflows/` has `docs.yml` and `publish.yml` only. `tests/` runs nowhere. | Every bug above would have been caught by a 10-line workflow. Releases publish to PyPI on tag with zero gating. | Yes — invisible to users |
| G2 | **Ingestion is fully serial.** One file at a time, one LLM call at a time, one embed call at a time. | 1,000 PDFs × (convert + LLM + embed) is hours. No `async`, no thread pool, no concurrency knob. | Yes — `workers: 1` default |
| G3 | **No retry/backoff anywhere.** Zero retries on LLM, embedding, or Qdrant calls. | Any 429 or transient 5xx fails a file permanently (and see B3/B4). | Yes — default on |
| G4 | **No document update path.** Dedup is by file hash; an edited document creates a *second* copy under a new hash with the old chunks still present. | Stale content is returned forever. There is no `delete_by_source` / `upsert`. | Yes — additive API |
| G5 | **No reranking.** Retrieval ends at vector/hybrid top-k. | Hybrid-then-rerank is the single biggest quality win in modern RAG. | Yes — opt-in `reranker:` block |
| G6 | **No evaluation harness.** No way to measure whether a config change helps. | "Production grade" is unmeasurable without recall@k / faithfulness on a golden set. | Yes — separate `ragwire.eval` |
| G7 | **No observability.** No token counts, no latency, no cost, no trace IDs. | Cannot answer "why was this answer wrong" or "what does ingestion cost". | Yes — no-op by default |
| G8 | **No CLI.** Every operation needs a Python script. | `ragwire ingest ./docs` / `ragwire query "..."` / `ragwire stats` would *reduce* time-to-first-success. | **Improves** setup |
| G9 | **No config validation.** `Config` is a raw dict with `.get()` defaults; typos (`chunk_sise`) are silently ignored. Missing env vars log a warning and leave a literal `${OPENAI_API_KEY}` as the API key ([config.py:105](ragwire/core/config.py#L105)). | Misconfiguration surfaces as a confusing 401 from a provider, not as a config error. | **Improves** setup |
| G10 | **`load_dotenv()` called inside `Config.__init__`** ([config.py:50](ragwire/core/config.py#L50)). | A library silently mutating `os.environ` of the host process surprises embedders. Make it opt-in. | Yes |
| G11 | **No chunk-level dedup.** `sha256_chunk` exists and is stored but never queried. | Boilerplate (headers, disclaimers, legends) repeated across filings floods top-k with duplicates. | Yes |
| G12 | **Local Qdrant path mode takes an exclusive lock** ([qdrant_store.py:67](ragwire/vectorstores/qdrant_store.py#L67)). Two `RAGWire` instances → lock error. | Any multi-worker server (gunicorn, uvicorn `--workers 2`) fails in path mode with no explanatory message. | Yes — detect and explain |
| G13 | **No `add_documents` batching or size guard.** A 10k-chunk document is one request. | Large documents blow request-size limits or time out mid-way, triggering B3. | Yes — internal default |
| G14 | **Tests are import smoke tests only.** No Qdrant fixture, no fake LLM, no ingestion test. | Coverage is nominal; `--cov` in `addopts` reports on code no test exercises. | Yes |

---

## 3. Action plan

Sequenced so that each phase is independently shippable.

### Phase 0 — Stop the bleeding (patch 1.3.3)
Small, no API change, no new dependency.

- [ ] B1 — `hmac.compare_digest`
- [ ] B11 — `force_recreate: false` in tracked `config.yaml`
- [ ] B12 — console handler honours `log_level`
- [ ] B15 — align truncation limit with documentation
- [ ] B5 — count zero-chunk files as failed with an actionable message
- [ ] B9 — normalize caller-supplied filters
- [ ] G1 — add `.github/workflows/test.yml` (pytest on 3.10–3.13, run before publish)

### Phase 1 — Correctness (minor 1.4.0)
- [ ] B3 — atomic per-file ingestion + completion marker; `reingest()` / `repair()`
- [ ] B4 — retry metadata extraction, then fail loudly or tag `metadata_status`
- [ ] B6 — derive payload index types from the declared schema, not a name allowlist
- [ ] B7 — narrow the exception handling around index creation
- [ ] B8 — union field keys across a sample, or use the configured schema
- [ ] B13 — embedding-dimension guard with an actionable error
- [ ] B10 — real dependency floors + a lockfile for CI
- [ ] B14 — enforce or delete `DocumentMetadata`; reconcile `fiscal_year` type
- [ ] G14 — real tests: fake LLM, in-memory Qdrant, full ingest→retrieve round trip

### Phase 2 — Scale (minor 1.5.0)
- [ ] G3 — retry/backoff on every network boundary (tenacity or hand-rolled)
- [ ] G2 — concurrent ingestion, `ingestion.workers` (default `1`)
- [ ] G13 — internal batching for `add_documents`
- [ ] G4 — `delete_by_source()` / `update_document()`; change detection on re-ingest
- [ ] G11 — chunk-hash dedup at write time
- [ ] G12 — detect path-mode lock contention and raise a clear message

### Phase 3 — Quality (minor 1.6.0)
- [ ] G5 — optional reranker (`reranker: {provider, model, top_n}`)
- [ ] G6 — `ragwire.eval`: golden-set recall@k, MRR, and a config A/B runner
- [ ] G7 — token/latency/cost callbacks; optional OpenTelemetry spans
- [ ] Contextual chunk headers (prepend document title + section to each chunk)
- [ ] Query rewriting / HyDE as opt-in retrieval strategies

### Phase 4 — Developer experience (minor 1.7.0)
- [ ] G8 — `ragwire` CLI: `init`, `ingest`, `query`, `stats`, `doctor`
- [ ] G9 — Pydantic-validated config with typo detection and fail-fast env resolution
- [ ] G10 — `Config(..., load_env=False)` opt-out
- [ ] `ragwire doctor` — checks Qdrant reachability, model availability, dimension
      match, and index health in one command
- [ ] Docker Compose quickstart (Qdrant + Ollama) as a single `docker compose up`

---

## 4. Design principles to hold the line on

These exist so "production grade" does not quietly destroy "simple setup."

1. **Zero-config defaults must stay zero-config.** Every feature added above lands
   as an optional config block. A `config.yaml` written for 1.3.2 must keep
   working unchanged.
2. **Fail loudly at init, silently never.** Dimension mismatch, unreachable
   Qdrant, missing model, unresolved `${ENV_VAR}` — all should raise at
   `RAGWire(...)` construction with a message naming the fix, not surface as a
   provider 401 twenty minutes into an ingest.
3. **Never swallow an exception without a log line.** B7 and B4 are the same
   mistake twice.
4. **One source of truth per concept.** Field types live in the schema, not in
   `_INTEGER_FIELDS`, not in `DocumentMetadata`, not in `FinancialMetadata`.
5. **Minimize global state.** No unconditional `load_dotenv`, no mutation of the
   root logger. (The warning-filter suppression in B2 is a deliberate exception.)
6. **Every bug fixed gets a regression test.** The bug list above is the initial
   test backlog.

---

## 5. Change log

| Date | Change |
|---|---|
| 2026-07-25 | Initial audit of v1.3.2 — 15 bugs, 14 gaps, 4-phase plan |
| 2026-07-25 | B2 (global warning filter) marked won't-fix — intentional design choice |
