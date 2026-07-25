# RAGWire Production Readiness Tracker

> Audit date: **2026-07-25** · Version audited: **1.3.2** · Branch: `main`
>
> Guiding constraint: **every change must keep the 3-line setup story intact.**
> `pip install ragwire` → `config.yaml` → `RAGWire("config.yaml")`. Nothing below
> may add a required step for the beginner path. Production features are opt-in.

**Status legend:** ☐ open · ◐ in progress · ☑ done · ✗ won't fix

> **Current status (2026-07-25):** all 14 actionable bugs fixed and covered by
> regression tests, plus G15. B2 is won't-fix by decision. Phases 0, 1, 2 and 3
> are complete; Phase 4 remains open. Suite: 217 tests, all passing, with no
> server, no LLM and no network needed.

---

## 1. Confirmed bugs

Ordered by blast radius. Each entry has a concrete failure scenario so it can be
turned into a regression test.

### B1: `compare_hashes()` raises `AttributeError` on every call ☑
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
in the pipeline calls it, which is exactly why it was never caught, and that is a
direct signal that the module has no unit test.

**Fix:** `import hmac` → `hmac.compare_digest(...)`. Add a test.

---

### B2: Global `warnings.filterwarnings("ignore")` on import ✗
**File:** [ragwire/utils/logging.py:14](ragwire/utils/logging.py#L14) · **Won't fix, intentional**

Importing `ragwire` mutates the host process's global warning filters. Kept
deliberately to suppress LangChain and dependency noise for the beginner path.
Logged here so it is not re-reported as a defect.

---

### B3: Failed ingestion leaves a permanently half-ingested document ☑
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
`failed: 1`. User re-runs and gets `skipped: 1`. The collection permanently
contains 150 of 400 chunks and every subsequent retrieval silently misses the back
half of the document. There is no command to recover from this state.

**Fix:** make ingestion atomic per file. Write chunks, then write a completion
marker (for example an `ingest_complete: true` payload on a sentinel point, or
upsert the `file_hash` record *last*). `file_hash_exists` must check the marker,
not any chunk. Add `rag.reingest(path, force=True)` and a `rag.repair()` that finds
file hashes whose stored chunk count differs from `total_chunks`.

---

### B4: LLM metadata failure is swallowed and becomes permanent ☑
**File:** [ragwire/core/pipeline.py:441-446](ragwire/core/pipeline.py#L441) · **Severity:** high

```python
try:
    llm_metadata = self.extract_metadata(text)
except Exception as e:
    logger.warning(f"LLM metadata extraction failed for {file_name}: {e}")
    # llm_metadata stays {}, and ingestion proceeds
```

A transient LLM timeout means the document is ingested with **zero semantic
metadata**. It still counts as `processed`, so the user sees a green run. Because
of file-hash dedup (B3), re-running ingestion **cannot fix it**, since the file is
skipped forever.

**Scenario:** Ollama is mid-model-swap during a 500-file batch. 40 files ingest
with no `company_name` or `fiscal_year`. Every metadata-filtered query silently
excludes them. The only recovery is wiping the collection.

**Fix:** retry with backoff (2 to 3 attempts), then either (a) fail the file so it
is retried on the next run, or (b) ingest it but tag `metadata_status: "failed"` so
it is discoverable and repairable. Surface a `metadata_failed` counter in
`IngestStats`.

---

### B5: Files producing zero chunks vanish from the stats ☑
**File:** [ragwire/core/pipeline.py:346-350](ragwire/core/pipeline.py#L346) · **Severity:** medium

```python
if chunks:
    self.vectorstore.add_documents(chunks)
    stats["chunks_created"] += len(chunks)
    stats["processed"] += 1
# no else branch
```

If `split_text` returns `[]`, which happens with a scanned PDF that has no text
layer, an empty `.txt`, or a password-protected DOCX that MarkItDown returns empty
for, the file is counted in `total` but in **none** of `processed`, `skipped` or
`failed`.

**Scenario:** `ingest_directory("data/")` on 100 files where 12 are image-only
scans returns `{total: 100, processed: 88, skipped: 0, failed: 0}`. The numbers
do not add up and the user has no way to learn which 12 files are missing. This is
the most common real-world RAG failure (scanned PDFs) and the framework hides it.

**Fix:** add an `else` branch that does `stats["failed"] += 1` with the error
`"no extractable text (possibly a scanned/image-only document)"`. Assert
`processed + skipped + failed == total` in tests.

---

### B6: Custom integer metadata fields get the wrong Qdrant index ☑
**File:** [ragwire/vectorstores/qdrant_store.py:283](ragwire/vectorstores/qdrant_store.py#L283) · **Severity:** high

```python
_INTEGER_FIELDS = {"chunk_index", "total_chunks", "fiscal_year"}
```

The integer-vs-keyword decision is a **hardcoded allowlist of the built-in
financial schema**. Custom schemas via `metadata.config_file` are a headline
feature, and any custom field declared `type: integer` (for example
`publication_year`, `revision`, `page_count`) falls through to
`PayloadSchemaType.KEYWORD`.

A keyword index does not index integer payload values. The subsequent
`create_payload_index` call fails or produces a useless index, and the failure is
swallowed by the bare `except` on line 311 (see B7). Downstream, `client.facet()`
returns nothing, so `get_field_values()` returns `[]` for that field, so
`_extract_filters_from_query` shows the LLM an empty value list and any filter it
produces matches zero points.

**Scenario:** an `examples/metadata.yaml`-style config with
`- name: publication_year, type: integer`. User asks *"papers from 2023"*.
`extract_filters` returns `{"publication_year": 2023}`, `retrieve()` builds a
valid filter, Qdrant returns **0 documents**, and the agent answers "no relevant
documents found" for a corpus that is full of them.

**Fix:** thread the declared field types from `MetadataExtractor.fields` or the
YAML into `create_payload_indexes`. Fall back to inferring the schema type from a
sampled payload value rather than a hardcoded name set.

---

### B7: Bare `except: pass` hides index-creation failures ☑
**File:** [ragwire/vectorstores/qdrant_store.py:311-312](ragwire/vectorstores/qdrant_store.py#L311) · **Severity:** medium

```python
except Exception:
    pass  # Already exists, safe to ignore
```

The comment assumes the only possible exception is "already exists." It also
swallows auth failures, connection resets, quota errors, and the wrong-schema-type
error from B6. Every payload index can silently fail to exist and the pipeline
reports success.

**Fix:** catch narrowly. Inspect the Qdrant error for the already-exists case,
`logger.debug` that, and `logger.warning` (or re-raise) everything else.

---

### B8: Metadata field discovery samples exactly one point ☑
**File:** [ragwire/vectorstores/qdrant_store.py:269-279](ragwire/vectorstores/qdrant_store.py#L269) · **Severity:** medium

```python
results, _ = self.client.scroll(collection_name=..., limit=1, ...)
metadata = payload.get("metadata", {})
return list(metadata.keys())
```

Field discovery reads **one arbitrary point** and treats its keys as the schema of
the entire collection. Two things break:

1. Fields that were `None` for that one document are absent from its payload, so
   they never get a payload index, and `get_field_values` fails for them.
2. Mixed-schema collections (built-in financial docs ingested first, then a custom
   schema) only ever expose whichever schema the sampled point belongs to.

**Scenario:** a collection where document #1 is an 8-K (`fiscal_quarter: None`).
`discover_metadata_fields()` omits `fiscal_quarter` entirely, so no index is
created, so filtering on quarter returns nothing across the whole collection.

**Fix:** sample N points (for example 100) and union the keys. Better still,
derive the field list from the configured schema (`self._filter_fields`) plus the
known system fields, which is authoritative and needs no network call.

---

### B9: Caller-supplied filters are not normalized; LLM-extracted ones are ☑
**File:** [ragwire/core/pipeline.py:648-651](ragwire/core/pipeline.py#L648) · **Severity:** medium

`_extract_filters_from_query` lowercases every extracted string value
([pipeline.py:609-613](ragwire/core/pipeline.py#L609)), and `MetadataExtractor.extract`
lowercases everything on write. But `retrieve(query, filters=...)` passes the
caller's dict straight to `_build_qdrant_filter` with **no normalization**.

**Scenario:** the documented agent pattern in
[examples/rag_agent.py:58](examples/rag_agent.py#L58) hands the LLM's tool-call
arguments directly to `rag.retrieve`. The agent writes
`filters={"company_name": "Apple Inc."}`. The stored value is `"apple inc."`.
Qdrant `MatchValue` is exact and case-sensitive, so the query returns
**0 results**. The agent concludes the corpus has nothing about Apple.

**Fix:** run caller-supplied filters through the same normalization as the
extracted path, using one shared `_normalize_filters()` helper called by both.

---

### B10: Dependency floors permit versions the code cannot run on ☑
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
| `response.text` as a **property** ([pipeline.py:604](ragwire/core/pipeline.py#L604)) | `langchain-core` 1.x. It was a *method* in 0.3.x and absent in 0.1.x. |
| `llm.with_structured_output()` ([extractor.py:108](ragwire/metadata/extractor.py#L108)) | `langchain-core` ≫ 0.1.0 |
| `RetrievalMode.HYBRID`, `FastEmbedSparse` ([qdrant_store.py:109](ragwire/vectorstores/qdrant_store.py#L109)) | `langchain-qdrant` ≫ 0.1.0 |

Locally installed `langchain-core` is 1.4.8, which is why it works on the author's
machine. A fresh resolve in a constrained environment (an existing project pinning
older LangChain) will install a permitted-but-broken combination and fail with
`AttributeError: 'function' object has no attribute 'strip'` deep inside filter
extraction. Optional extras are worse, since every one is `>=0.0.0`.

**Fix:** set real floors from what is actually tested (`langchain-core>=1.0`,
`langchain-qdrant>=0.2` and so on), add upper bounds on majors, and commit a
`requirements-lock.txt` used by CI.

---

### B11: Shipped `config.yaml` wipes the collection on every startup ☑
**File:** [config.yaml:52](config.yaml#L52) · **Severity:** high

```yaml
force_recreate: true  # Set true ONLY during testing to wipe and rebuild the collection
```

`config.yaml` is **tracked in git** and is the file every example loads
(`RAGWire("config.yaml")` in [examples/rag_agent.py:28](examples/rag_agent.py#L28)).
`config.example.yaml:82` and all 14 docs pages correctly say `false`. The one file
users actually run says `true`.

**Scenario:** a user ingests 2,000 documents, restarts their service the next day,
and `_initialize_vectorstore` calls `delete_collection()` before they issue a
single query. Total silent data loss on a plain restart, with no prompt and only
an INFO log line.

**Fix:** flip to `false` in the tracked `config.yaml`. Consider removing tracked
`config.yaml` entirely and having docs say "copy `config.example.yaml`", so there
is exactly one canonical template. Additionally, `force_recreate` should require a
second confirmation (an env var, or `RAGWire(..., allow_destructive=True)`) before
it deletes a non-empty collection.

---

### B12: `log_level: DEBUG` never reaches the console ☑
**File:** [ragwire/utils/logging.py:56](ragwire/utils/logging.py#L56), [:135](ragwire/utils/logging.py#L135) · **Severity:** low

```python
console_handler.setLevel(logging.INFO)   # hardcoded, ignores log_level
```

Both `setup_logging` and `setup_colored_logging` set the logger to the requested
level but pin the console handler to INFO. Setting `logging.level: "DEBUG"` in
config produces no additional console output, because DEBUG records are filtered
at the handler. Every `logger.debug` in the codebase (extracted metadata, payload
index creation, chunk details) is unreachable without a log file.

**Scenario:** a user debugging why metadata extraction returns nulls sets
`level: DEBUG`, sees no change, and has no way to inspect what the LLM returned.

**Fix:** `console_handler.setLevel(getattr(logging, log_level.upper()))`.

---

### B13: No embedding-dimension check against an existing collection ☑
**File:** [ragwire/core/pipeline.py:261-267](ragwire/core/pipeline.py#L261) · **Severity:** medium

When the collection already exists, RAGWire attaches to it without comparing the
stored `vector_size` to the dimension of the currently configured embedding model.

**Scenario:** a user switches `embeddings.model` from `nomic-embed-text` (768) to
`qwen3-embedding:0.6b` (1024). Init succeeds and logs "Using existing collection."
The first `add_documents` fails with a raw Qdrant dimension error, or worse, for
some client paths, queries return meaningless neighbours. The README's fix
("set `force_recreate: true` once") is only discoverable after the failure.

**Fix:** in `_initialize_vectorstore`, compare `len(embedding.embed_query("test"))`
against `get_collection_info().config.params.vectors.size` and raise a clear,
actionable error naming both dimensions and the exact remediation.

---

### B14: `DocumentMetadata` is declared, exported, and never used ☑
**File:** [ragwire/metadata/schema.py:13](ragwire/metadata/schema.py#L13) · **Severity:** low

`_process_document` builds a **raw dict** ([pipeline.py:453](ragwire/core/pipeline.py#L453))
and never validates it against `DocumentMetadata`. The model is exported in
`__all__` and tested in `tests/test_imports.py`, giving a false impression that
ingestion output is schema-validated. It also *contradicts* the real schema:
`DocumentMetadata.fiscal_year: Optional[List[int]]` versus
`FinancialMetadata.fiscal_year: Optional[int]`
([extractor.py:47](ragwire/metadata/extractor.py#L47)), while `_INTEGER_FIELDS`
indexes it as a scalar integer (B6). Three sources of truth, two of them wrong.

**Fix:** either validate chunk metadata through `DocumentMetadata` in
`_process_document` (and reconcile `fiscal_year`), or delete the model. Do not
ship an unenforced schema.

---

### B15: `extract()` truncates at 4,000 chars while docs promise 10,000 ☑
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

**Fix:** make the cap a constructor or config parameter (`extraction_char_limit`,
default 10000), and align the docstrings. Better still, derive the cap from the
model's context window rather than a magic number.

---

## 2. Production-grade gaps (missing capability, not defects)

| # | Gap | Why it blocks production | Keeps setup simple? |
|---|---|---|---|
| G1 ☑ | **No CI test job.** `.github/workflows/` has `docs.yml` and `publish.yml` only. `tests/` runs nowhere. | Every bug above would have been caught by a 10-line workflow. Releases publish to PyPI on tag with zero gating. | Yes, invisible to users |
| G2 ☑ | **Ingestion is fully serial.** One file at a time, one LLM call at a time, one embed call at a time. | 1,000 PDFs × (convert + LLM + embed) is hours. No `async`, no thread pool, no concurrency knob. | Yes, `workers: 1` default |
| G3 ☑ | **No retry/backoff anywhere.** Zero retries on LLM, embedding, or Qdrant calls. | Any 429 or transient 5xx fails a file permanently (and see B3/B4). | Yes, default on |
| G4 ☑ | **No document update path.** Dedup is by file hash; an edited document creates a *second* copy under a new hash with the old chunks still present. | Stale content is returned forever. There is no `delete_by_source` or `upsert`. | Yes, additive API |
| G5 ☑ | **No reranking.** Retrieval ends at vector/hybrid top-k. | Hybrid-then-rerank is the single biggest quality win in modern RAG. | Yes, opt-in `reranker:` block |
| G6 ☑ | **No evaluation harness.** No way to measure whether a config change helps. | "Production grade" is unmeasurable without recall@k or faithfulness on a golden set. | Yes, separate `ragwire.eval` |
| G7 | **No observability.** No token counts, no latency, no cost, no trace IDs. | Cannot answer "why was this answer wrong" or "what does ingestion cost". | Yes, no-op by default |
| G8 ◐ | **No CLI.** Every operation needs a Python script. | `ragwire ingest ./docs`, `ragwire query "..."` and `ragwire stats` would *reduce* time-to-first-success. | **Improves** setup |
| G9 | **No config validation.** `Config` is a raw dict with `.get()` defaults; typos (`chunk_sise`) are silently ignored. Missing env vars log a warning and leave a literal `${OPENAI_API_KEY}` as the API key ([config.py:105](ragwire/core/config.py#L105)). | Misconfiguration surfaces as a confusing 401 from a provider, not as a config error. | **Improves** setup |
| G10 | **`load_dotenv()` called inside `Config.__init__`** ([config.py:50](ragwire/core/config.py#L50)). | A library silently mutating `os.environ` of the host process surprises embedders. Make it opt-in. | Yes |
| G11 ☑ | **No chunk-level dedup.** `sha256_chunk` exists and is stored but never queried. | Boilerplate (headers, disclaimers, legends) repeated across filings floods top-k with duplicates. | Yes |
| G12 ☑ | **Local Qdrant path mode takes an exclusive lock** ([qdrant_store.py:67](ragwire/vectorstores/qdrant_store.py#L67)). Two `RAGWire` instances hit a lock error. | Any multi-worker server (gunicorn, uvicorn `--workers 2`) fails in path mode with no explanatory message. | Yes, detect and explain |
| G13 ☑ | **No `add_documents` batching or size guard.** A 10k-chunk document is one request. | Large documents blow request-size limits or time out mid-way, triggering B3. | Yes, internal default |
| G14 ☑ | **Tests are import smoke tests only.** No Qdrant fixture, no fake LLM, no ingestion test. | Coverage is nominal; `--cov` in `addopts` reports on code no test exercises. | Yes |

### G15: Payload indexes do not work in local (path) mode ☑

Surfaced while writing the B6 regression tests. `QdrantClient(path=...)`, the
mode used whenever `vectorstore.url` is not an HTTP URL, emits:

> *"Payload indexes have no effect in the local Qdrant."*

Payload indexes are what the facet API needs, so on local storage
`get_field_values()` returned empty lists, which meant `auto_filter` and
`get_filter_context()` showed the LLM no stored values and silently degraded.
The library gave no indication that a headline feature was inert.

**Fixed** by making it work rather than only warning about it. A warning would
still leave metadata filtering broken on the zero-setup path, which is the one
most beginners use:

- `QdrantStore.is_local` is set at init, with one INFO line naming the tradeoff
  and pointing at the Docker one-liner.
- `create_payload_indexes` skips entirely in local mode, so the per-field
  `UserWarning` spam is gone.
- `get_field_values` falls back to `_scan_field_values`, which scrolls points
  and collects distinct values, flattening list-valued fields. It stops as soon
  as every field has `limit` values, so the common case reads one page, and
  caps at `_SCAN_LIMIT` (10,000) points with a warning if it hits the cap.

Server mode is unchanged and still uses the facet API.

---

## 3. Action plan

Sequenced so that each phase is independently shippable.

### Phase 0: Stop the bleeding (patch 1.3.3) ☑
Small, no API change, no new dependency.

- [x] B1: `hmac.compare_digest`
- [x] B11: `force_recreate: false` in tracked `config.yaml`
- [x] B12: console handler honours `log_level`
- [x] B15: align truncation limit with documentation
- [x] B5: count zero-chunk files as failed with an actionable message
- [x] B9: normalize caller-supplied filters
- [x] G1: add `.github/workflows/test.yml` (pytest on 3.10 to 3.13, run before publish)

### Phase 1: Correctness (minor 1.4.0) ☑
- [x] B3: atomic per-file ingestion plus completion marker; `reingest()` and `repair()`
- [x] B4: retry metadata extraction, then fail loudly or tag `metadata_status`
- [x] B6: derive payload index types from the declared schema, not a name allowlist
- [x] B7: narrow the exception handling around index creation
- [x] B8: union field keys across a sample, or use the configured schema
- [x] B13: embedding-dimension guard with an actionable error
- [x] B10: real dependency floors plus a lockfile for CI
- [x] B14: enforce or delete `DocumentMetadata`; reconcile `fiscal_year` type
- [x] G14: real tests with a fake LLM, in-memory Qdrant, and a full ingest→retrieve round trip

### Phase 2: Scale (shipped in 1.4.1) ☑
- [x] G3: retry with exponential backoff on every network boundary
- [x] G2: concurrent ingestion via `ingestion.workers` (default `1`)
- [x] G13: internal batching for `add_documents` via `ingestion.batch_size` (default `64`)
- [x] G4: `delete_by_source()` and `delete_document()`; change detection on re-ingest
- [x] G11: chunk-hash dedup at write time via `ingestion.dedup_chunks` (default off)
- [x] G12: detect path-mode lock contention and raise a message naming the fix

Design note: preparation (load, split, extract metadata) is concurrent, while
writes stay sequential in the parent. Mutation therefore happens in exactly one
place, so rollback and stats stay correct without locking. `ThreadPoolExecutor.map`
preserves input order, so logs and counters are deterministic regardless of which
document finishes first. `retry_call` re-raises `TypeError`, `AttributeError` and
`ImportError` immediately instead of retrying, so genuine bugs surface at once
rather than after three backoff delays.

### Phase 3: Quality and reach (minor 1.5.0) ☑
- [x] G5: optional reranker via `retriever.rerank`, local `cross_encoder` default plus hosted `cohere`
- [x] G6: `ragwire.eval` with golden-set recall@k, MRR, hit rate, precision and a `sweep()` A/B runner
- [x] `rag.query()` and `rag.aquery()`: grounded answers with citations, refusal and a groundedness score
- [x] MCP server (`ragwire mcp serve`) exposing four tools to Claude Desktop, Claude Code and Cursor
- [x] Source connectors (`local`, `s3`) plus `rag.sync()` reconciliation including deletions
- [x] G8 (partial): `ragwire` CLI with `ingest`, `sync`, `eval` and `mcp serve`
- [ ] G7: token, latency and cost callbacks; optional OpenTelemetry spans
- [ ] Contextual chunk headers (prepend document title and section to each chunk)
- [ ] Query rewriting and HyDE, which the `retriever.rerank` block was shaped to accept

Design notes:

**Reranking is off unless configured, and free when on.** The default provider
is a local cross-encoder, so second-stage retrieval never becomes a paid
feature. The model loads on first use rather than at construction, so an
ingestion-only script never downloads it, while the package check stays eager
so a misconfiguration fails at startup rather than on the first query.

**Refusal is a return value, not an exception.** `query()` returns an `Answer`
with `refused=True` when the sources do not support one. An unanswerable
question is an ordinary outcome, and forcing callers into a `try` block to
handle it would push them toward catching and ignoring it. `Answer.__bool__`
reflects the same thing so `if not answer:` reads correctly.

**`confidence` is citation coverage, and is documented as such.** Calling it
accuracy would be a lie a user could act on: a fully cited answer drawn from a
chunk that happens to be wrong still scores 1.0.

**Sync suppresses deletion whenever a source cannot be trusted.** A source that
fails to list, or lists zero files, cancels the deletion pass for the whole run.
"The bucket returned nothing" and "every object was deleted" are
indistinguishable from outside, and acting on the wrong reading empties the
collection. Ingestion still proceeds; only the destructive half is held back.

**Empty MCP results say so out loud.** An agent reads silence as a broken tool
and retries indefinitely, so `search_documents` returns an explicit statement
plus a pointer to `get_filter_context`. Likewise a refusal from
`answer_question` instructs the agent not to fill the gap from its own
knowledge, without which the collection boundary quietly disappears at exactly
the moment it matters.

### Phase 4: Developer experience (minor 1.6.0)
- [ ] G8 (remainder): `ragwire init`, `query`, `stats` and `doctor` subcommands
- [ ] G9: Pydantic-validated config with typo detection and fail-fast env resolution
- [ ] G10: `Config(..., load_env=False)` opt-out
- [ ] Google Drive source connector (deliberately deferred: needs an interactive
      OAuth flow and token storage, and a half-built one is worse than the
      documented `REGISTRY.register` extension point)
- [ ] `ragwire doctor`, which checks Qdrant reachability, model availability,
      dimension match, and index health in one command
- [ ] Docker Compose quickstart (Qdrant plus Ollama) as a single `docker compose up`

---

## 4. Design principles to hold the line on

These exist so "production grade" does not quietly destroy "simple setup."

1. **Zero-config defaults must stay zero-config.** Every feature added above lands
   as an optional config block. A `config.yaml` written for 1.3.2 must keep
   working unchanged.
2. **Fail loudly at init, silently never.** Dimension mismatch, unreachable
   Qdrant, missing model, unresolved `${ENV_VAR}`: all should raise at
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
| 2026-07-25 | Initial audit of v1.3.2: 15 bugs, 14 gaps, 4-phase plan |
| 2026-07-25 | B2 (global warning filter) marked won't-fix, an intentional design choice |
| 2026-07-25 | Phase 0 shipped: B1, B5, B9, B11, B12, B15 plus the CI test workflow (`85bbf68`, `13f6a53`) |
| 2026-07-25 | Phase 1 shipped: B3, B4, B6, B7, B8, B10, B13, B14 plus 31 regression tests (`13f6a53`, `bb3019e`) |
| 2026-07-25 | New API: `reingest_documents()`, `delete_document()`, `IngestStats.metadata_failed`, `metadata_status` payload field |
| 2026-07-25 | G15 logged: payload indexes are inert in local path mode, silently disabling metadata filtering |
| 2026-07-25 | G15 fixed: local mode now collects field values by scanning points; server mode still uses facets |
| 2026-07-25 | v1.4.0 released to PyPI |
| 2026-07-25 | Phase 2 shipped: G2, G3, G4, G11, G12, G13 plus 17 ingestion tests (`84a88fe`). Suite now 58 tests. |
| 2026-07-25 | New config block `ingestion:` with `workers`, `batch_size`, `retries`, `replace_changed`, `dedup_chunks`; new `IngestStats.replaced`; every chunk now carries `content_hash` |
| 2026-07-25 | Documentation pass: `docs/setup.md` rewritten after finding it configured Ollama without ever telling the reader to install it or pull the models |
| 2026-07-25 | Em and en dashes removed repo-wide (653 occurrences, 66 files), each rewritten in context rather than substituted |
| 2026-07-25 | Fixed a doc/schema contradiction: `fiscal_year` was documented as `list[int]` in README, `docs/metadata.md`, `llms.txt`, `llms-full.txt`, `AGENT.md` and the `json_schema_extra` example, while `FinancialMetadata` declares `Optional[int]` |
| 2026-07-25 | Version bumped to 1.4.1, carrying Phase 2 plus the documentation and dash pass |
| 2026-07-25 | Publishing now triggers on pushing a `v*` tag rather than on creating a GitHub release, with a guard that fails the build when the tag and `pyproject.toml` disagree |
| 2026-07-25 | Phase 3 shipped across five commits: reranking (`fe291f1`), `ragwire.eval` (`4db9a40`), `rag.query()` (`b75132e`), MCP server and CLI (`b8704ed`), source sync (`2954395`). Suite grew from 58 to 217 tests. |
| 2026-07-25 | New config blocks: `retriever.rerank`, `generation`, `sources`. New extras: `rerank`, `cohere`, `mcp`, `s3`. New console script: `ragwire`. |
| 2026-07-25 | Found and fixed while building: the CI workflow installed only `pytest` and `pytest-cov` and would have failed on the new async tests; the `ragwire ingest` command read `IngestStats` keys that do not exist |
