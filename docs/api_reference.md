# API Reference

All public APIs are importable directly from `ragwire`:

```python
from ragwire import RAGWire, MarkItDownLoader, get_embedding, QdrantStore, ...
```

---

## Core API

These are the primary user-facing APIs. Most applications only need these.

---

### RAGWire

The main orchestrator. Handles the full pipeline from config loading to ingestion and retrieval.

```python
from ragwire import RAGWire
```

#### `RAGWire(config_path)`

Initialize the pipeline from a YAML config file.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `config_path` | `str` | Yes | Path to `config.yaml` |

**Raises:**

- `FileNotFoundError`: config file not found
- `ValueError`: missing required config keys (e.g. `llm.model`)

```python
rag = RAGWire("config.yaml")
```

---

#### `rag.ingest_documents(file_paths)`

Ingest a list of documents into the vector store. Skips files already ingested (SHA256 deduplication).

| Parameter | Type | Required | Description |
|---|---|---|---|
| `file_paths` | `list[str]` | Yes | List of file paths to ingest |

**Returns:** `dict`

```python
{
    "total": 3,           # Total files submitted
    "processed": 2,       # Successfully ingested
    "skipped": 1,         # Already in vector store (duplicate)
    "failed": 0,          # Failed to load or process
    "chunks_created": 84, # Total chunks added to Qdrant
    "metadata_failed": 0, # Ingested, but LLM metadata extraction failed
    "replaced": 0,        # Changed files whose old chunks were removed first
    "errors": []          # List of {"file": ..., "error": ...} dicts
}
```

`processed + skipped + failed` always equals `total`. If a document is missing
from your results, that identity is the first thing to check.

Two counters deserve attention:

- **`metadata_failed`** counts documents that were stored but carry no
  LLM-extracted metadata, usually because the LLM was unreachable. They are
  included in `processed` and their text is searchable, but they match no
  metadata filter until re-ingested. Each such chunk is tagged
  `metadata_status: "failed"`. Fix the LLM, then call `reingest_documents()`.
- **`replaced`** counts files whose content changed since a previous ingest.
  Their older chunks were deleted before the new ones were written, so the
  collection holds exactly one version. Controlled by `ingestion.replace_changed`.

```python
stats = rag.ingest_documents([
    "data/Apple_10k_2025.pdf",
    "data/Microsoft_10k_2025.pdf",
])
print(f"Processed: {stats['processed']}, Chunks: {stats['chunks_created']}")
```

A progress bar (`tqdm`) is shown automatically while ingestion runs.

---

#### `rag.reingest_documents(file_paths)`

Force re-ingestion, replacing whatever is already stored for those files. Every
existing chunk for each file is deleted first, so this bypasses the SHA256
deduplication that makes `ingest_documents()` a no-op on unchanged files.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `file_paths` | `list[str]` | Yes | Paths to re-ingest |

**Returns:** `dict`, the same stats dict as `ingest_documents()`

Use it in two situations:

1. **Recovering documents ingested without metadata.** If a run reported
   `metadata_failed` above zero, those documents are invisible to filtered
   retrieval until re-ingested.
2. **A document changed on disk** and `ingestion.replace_changed` is off, so the
   old version is still stored alongside the new one.

```python
stats = rag.ingest_documents(["data/report.pdf"])

if stats["metadata_failed"]:
    # The LLM was down. Fix it, then repair only the affected documents.
    rag.reingest_documents(["data/report.pdf"])
```

---

#### `rag.delete_document(file_path)`

Remove every chunk of a document from the collection.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `file_path` | `str` | Yes | Path to the source file |

**Returns:** `int`, the number of chunks deleted. Returns `0` if the document was
never ingested.

!!! warning "The file must still exist on disk"
    Chunks are identified by the SHA256 hash of the file's contents, so the file
    has to be readable for its hash to be computed. Delete from the collection
    before deleting from disk, not after.

```python
removed = rag.delete_document("data/old_report.pdf")
print(f"Removed {removed} chunks")
```

---

#### `rag.ingest_directory(directory, recursive, extensions)`

Ingest all supported documents from a directory. Internally calls `ingest_documents()`.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `directory` | `str` | Yes | n/a | Path to the directory |
| `recursive` | `bool` | No | `False` | Search subdirectories |
| `extensions` | `list[str]` | No | loader config | File extensions to include |

**Returns:** `dict`, the same stats dict as `ingest_documents()`

```python
# Ingest all PDFs/DOCX in a folder
stats = rag.ingest_directory("data/")

# Recursively include subdirectories
stats = rag.ingest_directory("data/", recursive=True)

# Only specific extensions
stats = rag.ingest_directory("data/", extensions=[".pdf"])
```

---

#### `rag.retrieve(query, top_k, filters, rerank)`

Retrieve the most relevant chunks for a query.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `query` | `str` | Yes | n/a | Search query |
| `top_k` | `int` | No | config value | Number of results to return |
| `filters` | `dict` | No | `None` | Metadata filters (see [Metadata & Filtering](metadata.md)) |
| `rerank` | `bool` | No | config value | Override reranking for this call. `False` skips it, `True` requires it. |

**Returns:** `list[Document]`

Each `Document` has:

- `doc.page_content`: the chunk text
- `doc.metadata`: a dict with all metadata fields (see [Metadata Schema](metadata.md#metadata-schema))
- `doc.metadata["rerank_score"]`: the relevance score, present only when reranking ran

**Reranking behaviour:** with a `retriever.rerank` block configured, `retrieve()` fetches `fetch_k` candidates from the vector store, scores each one against the query, and returns the best `top_k`. Without one it returns the vector store's own `top_k` unchanged. Passing `rerank=True` with no reranker configured raises `ValueError` rather than silently returning unreranked results. See [Reranking](cookbook/reranking.md).

**Filter behaviour:**

- If `filters` is passed → used as-is, no LLM call (always, regardless of `auto_filter` setting)
- If `filters` is not passed and `auto_filter: true` in config → LLM extracts filters from the query
- If `filters` is not passed and `auto_filter: false` (default) → no filtering, pure semantic search

**When to use auto-filter vs explicit filters:** Use explicit filters in programmatic pipelines where you control the inputs (faster, zero LLM overhead). Enable `auto_filter` in simple user-facing chatbots. For agents, keep `auto_filter: false` and use `rag.extract_filters(query)` to give the agent full control over whether and how to apply filters.

```python
# Explicit filters, so LLM extraction is skipped
results = rag.retrieve(
    "What is the net income?",
    top_k=5,
    filters={"company_name": "apple", "fiscal_year": 2025}
)

# auto_filter: true in config, so the LLM extracts
# {"company_name": "apple", "fiscal_year": 2025}
results = rag.retrieve("What is Apple's net income for 2025?")

# auto_filter: false (default) gives pure semantic search, no filter extraction
results = rag.retrieve("What is Apple's net income for 2025?")

for doc in results:
    print(doc.metadata.get("company_name"))
    print(doc.page_content[:300])
```

---

#### `rag.sync(sources, delete_missing, dry_run)`

Reconcile the collection against its configured sources: ingest what is new, replace what changed, remove what is gone. See [Sync Sources](cookbook/syncing_sources.md).

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `sources` | `list[Source]` | No | config value | Sources to reconcile against. Uses the `sources` config block when not given. |
| `delete_missing` | `bool` | No | `true` | Remove documents no source lists any more |
| `dry_run` | `bool` | No | `false` | Report what would happen without writing or deleting |

**Returns:** `SyncStats`

| Key | Description |
|---|---|
| `listed` | Files the sources reported |
| `processed` | Documents newly written |
| `skipped` | Documents already present and unchanged |
| `replaced` | Documents whose content changed |
| `deleted` | Documents removed because no source lists them |
| `deleted_chunks` | Chunks removed by those deletions |
| `failed` | Files that could not be processed |
| `chunks_created` | Chunks written |
| `warnings` | Deletions held back for safety, with the reason |
| `errors` | `{"file": ..., "error": ...}` entries |

**Raises:** `ValueError` if no sources are configured or passed.

!!! warning "Deletion is suppressed when a source cannot be trusted"
    If any source fails to list, or lists zero files, sync deletes nothing that run and records why in `warnings`. An unreachable bucket and an emptied bucket look identical from the outside, and acting on the wrong reading would empty your collection. Ingestion still runs.

```python
stats = rag.sync(dry_run=True)   # look first
stats = rag.sync()               # then commit
```

---

#### `rag.query(question, top_k, filters, rerank)`

Answer a question from the collection, with citations. See [Answer Questions](cookbook/answering_questions.md).

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `question` | `str` | Yes | n/a | The question to answer |
| `top_k` | `int` | No | config value | How many chunks to ground the answer in |
| `filters` | `dict` | No | `None` | Metadata filters. Extracted automatically when `auto_filter: true`. |
| `rerank` | `bool` | No | config value | Override reranking, as in `retrieve()` |

**Returns:** `Answer`

| Attribute | Type | Description |
|---|---|---|
| `text` | `str` | The answer, or an explanation of why none could be given |
| `citations` | `list[Citation]` | The sources actually cited, in order of first reference |
| `documents` | `list[Document]` | Everything retrieved, cited or not |
| `filters_used` | `dict` \| `None` | Filters applied during retrieval |
| `refused` | `bool` | True when the sources did not support an answer |
| `confidence` | `float` | Fraction of sentences carrying a citation. Groundedness, not accuracy. |
| `sources` | `list[str]` | Distinct files cited |
| `formatted()` | `str` | The answer with a numbered source list appended |
| `to_dict()` | `dict` | Serialised for logging or an API response |

Each `Citation` has `index`, `source`, `text`, `snippet`, `metadata` and `score` (the rerank score, when one exists).

An `Answer` is falsy when it was refused, so `if not answer:` handles the unanswerable case. `query()` never raises for a question the documents cannot answer.

```python
answer = rag.query("What was Apple's net income in fiscal 2025?")

if not answer:
    print("Not in the collection")
else:
    print(answer.formatted())
    print(f"grounded: {answer.confidence:.0%}")
```

---

#### `rag.aquery(question, top_k, filters, rerank)`

Async version of `query()`. Same parameters, same return type. Only the LLM call is awaited; retrieval runs synchronously.

```python
answer = await rag.aquery("What was net income?")
```

---

#### `rag.hybrid_search(query, k, filters)`

Perform hybrid search combining dense (semantic) and sparse (keyword) vectors. Requires `use_sparse: true` in config.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `query` | `str` | Yes | n/a | Search query |
| `k` | `int` | No | `5` | Number of results |
| `filters` | `dict` | No | `None` | Metadata filters |

**Returns:** `list[Document]`

!!! warning "Hybrid search requires sparse vectors"
    `hybrid_search()` only performs true hybrid (dense + sparse) search when **both** conditions are met:

    1. `use_sparse: true` in `config.yaml`, so the collection is created with sparse vector support
    2. `pip install fastembed`, which supplies the sparse encoder

    If either is missing, the call silently falls back to **dense-only similarity search**. There is no error raised.
    If your collection was created with `use_sparse: false`, you must set `force_recreate: true` and re-ingest to enable hybrid search.

**Choosing between `retrieve()` and `hybrid_search()`:**

| | `retrieve()` | `hybrid_search()` |
|---|---|---|
| Search type | Whatever is set in `config.yaml` (`similarity`, `mmr`, or `hybrid`) | Always hybrid (dense + sparse), regardless of config |
| Auto-filter | Only when `auto_filter: true` in config (default `false`) | Same: respects the `auto_filter` setting |
| `top_k` default | From `config.yaml` | `k=5` parameter |
| Typical use | Primary method for all RAG flows | Override to force hybrid on a single call |

If your `config.yaml` already has `search_type: "hybrid"`, both methods produce identical results. Use `hybrid_search()` only when your config is set to `similarity` or `mmr` and you want to force hybrid for a specific call.

```python
# Use retrieve() in most cases. It honours the configured search type.
results = rag.retrieve("Apple revenue fiscal 2025", top_k=5)

# Use hybrid_search() to force hybrid regardless of config
results = rag.hybrid_search(
    "Apple revenue fiscal 2025",
    k=5,
    filters={"company_name": "apple"}
)
```

---

#### `rag.extract_filters(query)`

Extract metadata filters from a natural language query without triggering retrieval. Returns the raw extracted dict so an agent can inspect, adjust, or discard before passing to `retrieve()`.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `query` | `str` | Yes | Natural language query |

**Returns:** `dict` of extracted filters, or `None` if nothing was extracted.

!!! note
    This method always runs regardless of the `auto_filter` config setting. It gives agents explicit control: call it manually, decide what to do, then pass the result to `retrieve(filters=...)`.

```python
# Agent workflow with full control over filters
filters = rag.extract_filters("muscle building studies from 2023")
# → {"research_focus": "muscle building", "publication_year": 2023}

# Agent validates against stored values
stored = rag.get_field_values(rag.filter_fields)
if filters.get("research_focus") not in stored.get("research_focus", []):
    filters.pop("research_focus")  # drop uncertain filter, rely on semantic search

results = rag.retrieve("muscle building studies from 2023", filters=filters)
```

---

#### `rag.get_filter_context(query, limit)`

Build a ready-made markdown prompt block for an agent. It contains the available metadata fields, their stored values, the filters extracted from the current query, and instructions for the agent on how to act on them. Append or prepend to your agent's task prompt.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `query` | `str` | Yes | n/a | Natural language query |
| `limit` | `int` | No | `50` | Max stored values to show per field |

**Returns:** `str`, a formatted markdown block ready to inject into an agent prompt.

```python
filter_context = rag.get_filter_context("muscle building studies from 2023")
agent_prompt = filter_context + "\n\n" + your_task_prompt
```

The returned block looks like:

```
## RAGWire Filter Context

### Available Metadata Fields and Stored Values
- **research_focus**: ["muscle-growth", "endurance", "recovery", ...]
- **publication_year**: [2022, 2023, 2024]
- **authors**: ["john smith", "jane doe", ...]

### Extracted Filters from Query
- **research_focus**: `muscle building`
- **publication_year**: `2023`

### Instructions
1. Review the extracted filters above.
2. If an extracted value does not match or closely relate to any stored value, adjust or drop that filter.
3. If the query has no clear metadata intent, pass an empty dict {} as filters.
4. Pass the final filters dict to the retrieval tool as filters=.
```

!!! note "Typical agent workflow"
    Use `get_filter_context()` to give the agent full situational awareness. The agent can then call `rag.retrieve(query, filters=adjusted_filters)` with a well-informed decision on which filters to apply.

---

#### `rag.filter_fields`

Property. Returns the metadata fields used for filtering and auto-filter extraction, meaning the LLM-extracted semantic fields only. System fields like `file_hash`, `chunk_id`, `source`, `chunk_index`, `created_at` are excluded.

Use this when building dynamic filter prompts for an LLM agent. Using `discover_metadata_fields()` instead would include system fields that have no value as filters.

```python
fields = rag.filter_fields
# Default: ['company_name', 'doc_type', 'fiscal_quarter', 'fiscal_year']
# Custom:  whatever fields are defined in your metadata.yaml

values = rag.get_field_values(fields)
# → {'company_name': ['apple', 'microsoft'], 'doc_type': ['10-k'], ...}
```

---

#### `rag.discover_metadata_fields()`

Return **all** metadata field names present in the collection, including system fields. It scrolls a single point, so it stays fast regardless of collection size.

Use this for collection inspection or debugging. For building filter prompts, use `rag.filter_fields` instead.

**Returns:** `list[str]`

```python
fields = rag.discover_metadata_fields()
print(fields)
# ['company_name', 'doc_type', 'fiscal_year', 'fiscal_quarter',
#  'file_name', 'file_type', 'file_hash', 'chunk_id', 'chunk_index', ...]
```

---

#### `rag.get_field_values(fields, limit)`

Return unique values for one or more metadata fields using Qdrant's facet API. Results are ordered by frequency (most common values first).

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `fields` | `str \| list[str]` | Yes | n/a | Field name or list of field names |
| `limit` | `int` | No | `50` | Max unique values to return per field. Increase for high-cardinality fields (e.g. `file_name`). |

**Returns:**
- `list` if `fields` is a `str`
- `dict[str, list]` if `fields` is a `list`

```python
# Single field returns a list of up to 50 unique values
rag.get_field_values("company_name")
# → ['apple', 'microsoft', 'google']

# Multiple fields return a dict
rag.get_field_values(["company_name", "doc_type"])
# → {'company_name': ['apple', 'microsoft', 'google'], 'doc_type': ['10-k', '10-q']}

# High-cardinality field, so raise the limit
rag.get_field_values("file_name", limit=200)
# → ['Apple_10k_2025.pdf', 'Microsoft_10k_2025.pdf', ...]

# Typical agent workflow: use filter_fields, not discover_metadata_fields()
values = rag.get_field_values(rag.filter_fields)
results = rag.retrieve("revenue", filters={"company_name": values["company_name"][0]})
```

---

#### `rag.extract_metadata(text)`

Extract structured metadata from text using the configured LLM.

Automatically passes stored collection values so the LLM reuses existing entity names (e.g. `"apple inc."`) rather than extracting inconsistent variants (`"apple"`, `"Apple Inc."`). This grounding is applied automatically, so you do not need to pass stored values yourself.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `text` | `str` | Yes | Document text to extract metadata from (first 10,000 chars used) |

**Returns:** `dict`

```python
metadata = rag.extract_metadata(open("report.txt").read())
print(metadata)
# {'company_name': 'apple inc.', 'doc_type': '10-k', 'fiscal_quarter': None, 'fiscal_year': 2025}
```

---

#### `rag.get_stats()`

Get statistics about the current collection.

**Returns:** `dict`

```python
{
    "collection_name": "financial_docs",
    "total_documents": 420,   # Total chunks in Qdrant
    "vector_size": 768,       # Embedding dimension
    "indexed": 420            # Number of indexed vectors
}
```

```python
stats = rag.get_stats()
print(f"Collection: {stats['collection_name']}, Chunks: {stats['total_documents']}")
```

---

---

## Config Reference: `llm` and `embeddings`

All parameters below are set in `config.yaml` and read automatically by `RAGWire` at startup.

---

### `llm` section

Controls the LLM used for metadata extraction (and filter extraction during retrieval).

| Key | Required | Default | Description |
|---|---|---|---|
| `provider` | Yes | n/a | `ollama`, `openai`, `openrouter`, `google`, `groq`, `anthropic` |
| `model` | Yes | n/a | Model name (e.g. `qwen3.5:9b`, `gpt-4o-mini`, `poolside/laguna-m.1:free`) |
| `base_url` | Ollama only | `http://localhost:11434` | Ollama server URL |
| `num_ctx` | Ollama only | LangChain default | Context window size. Set this only to override the default. |
| `api_key` | Google / Groq / Anthropic / OpenRouter | n/a | API key (or use `${ENV_VAR}` syntax) |

!!! note "OpenAI"
    OpenAI reads `OPENAI_API_KEY` from the environment automatically, so no `api_key` field is needed in config.

!!! note "OpenRouter"
    Uses the dedicated `ChatOpenRouter` integration (`pip install "ragwire[openrouter]"`, Python ≥ 3.10). Reads `OPENROUTER_API_KEY` from the environment automatically; `api_key` in config is optional.

```yaml
# Ollama
llm:
  provider: "ollama"
  model: "qwen3.5:9b"
  base_url: "http://localhost:11434"
  num_ctx: 16384

# OpenAI
llm:
  provider: "openai"
  model: "gpt-4o-mini"

# OpenRouter (free-tier models available)
llm:
  provider: "openrouter"
  model: "poolside/laguna-m.1:free"
  api_key: "${OPENROUTER_API_KEY}"

# Google Gemini
llm:
  provider: "google"
  model: "gemini-2.5-flash"
  api_key: "${GOOGLE_API_KEY}"

# Groq
llm:
  provider: "groq"
  model: "llama-3.3-70b-versatile"
  api_key: "${GROQ_API_KEY}"

# Anthropic
llm:
  provider: "anthropic"
  model: "claude-haiku-4-5-20251001"
  api_key: "${ANTHROPIC_API_KEY}"
```

---

### `embeddings` section

Controls the embedding model used to encode documents and queries into vectors.

| Key | Required | Default | Description |
|---|---|---|---|
| `provider` | Yes | n/a | `ollama`, `openai`, `openrouter`, `google`, `huggingface`, `fastembed` |
| `model` | Most providers | provider default | Embedding model name |
| `base_url` | Ollama only | `http://localhost:11434` | Ollama server URL |
| `num_ctx` | Ollama only | LangChain default | Context window size. Set this only to override the default. |
| `api_key` | Google / OpenRouter | n/a | API key (or use `${ENV_VAR}` syntax) |
| `batch_size` | OpenRouter only | `100` | Inputs sent per embedding request |
| `dimensions` | OpenRouter only | model default | Output dimensionality (only if the model supports it) |
| `model_name` | HuggingFace / FastEmbed only | see below | Model identifier (uses `model_name` key, not `model`) |
| `model_kwargs` | HuggingFace only | `{}` | Passed to the HuggingFace model constructor (e.g. `{"device": "cpu"}`) |
| `encode_kwargs` | HuggingFace only | `{}` | Passed to the encode call (e.g. `{"normalize_embeddings": true}`) |

**Default models per provider:**

| Provider | Default model |
|---|---|
| `ollama` | `nomic-embed-text` |
| `openai` | `text-embedding-3-small` |
| `openrouter` | `nvidia/llama-nemotron-embed-vl-1b-v2:free` |
| `google` | `models/embedding-001` |
| `huggingface` | `sentence-transformers/all-MiniLM-L6-v2` |
| `fastembed` | `BAAI/bge-small-en-v1.5` |

```yaml
# Ollama
embeddings:
  provider: "ollama"
  model: "nomic-embed-text"
  base_url: "http://localhost:11434"
  num_ctx: 16384

# OpenAI
embeddings:
  provider: "openai"
  model: "text-embedding-3-small"

# OpenRouter (free-tier models available)
embeddings:
  provider: "openrouter"
  model: "nvidia/llama-nemotron-embed-vl-1b-v2:free"
  api_key: "${OPENROUTER_API_KEY}"

# Google Gemini
embeddings:
  provider: "google"
  model: "models/gemini-embedding-001"
  api_key: "${GOOGLE_API_KEY}"

# HuggingFace (local)
embeddings:
  provider: "huggingface"
  model_name: "sentence-transformers/all-MiniLM-L6-v2"
  model_kwargs:
    device: "cpu"
  encode_kwargs:
    normalize_embeddings: true

# FastEmbed (local, sparse-capable)
embeddings:
  provider: "fastembed"
  model_name: "BAAI/bge-small-en-v1.5"
```

---

### `retriever` section

Controls retrieval behaviour.

| Key | Required | Default | Description |
|---|---|---|---|
| `search_type` | No | `"hybrid"` | `"similarity"` \| `"mmr"` \| `"hybrid"` (hybrid requires `use_sparse: true`) |
| `top_k` | No | `5` | Number of results returned by `retrieve()` |
| `auto_filter` | No | `false` | If `true`, LLM automatically extracts metadata filters from every query passed to `retrieve()` / `hybrid_search()`. If `false`, no filter extraction happens unless `filters=` is passed explicitly or `rag.extract_filters()` is called manually. |
| `rerank` | No | absent | Optional second-stage reranking. See below. |

```yaml
retriever:
  search_type: "hybrid"
  top_k: 5
  auto_filter: false   # set true to enable automatic filter extraction from queries
```

!!! note "Agent use case"
    Keep `auto_filter: false` when an agent is driving retrieval. Use `rag.extract_filters(query)` to let the agent inspect and adjust filters before calling `retrieve(filters=...)`.

#### `retriever.rerank` subsection

Omit the block entirely and no reranking happens, which is the default. Adding it is enough to switch reranking on.

| Key | Required | Default | Description |
|---|---|---|---|
| `provider` | No | `"cross_encoder"` | `"cross_encoder"` (local, no API key) \| `"cohere"` (hosted) |
| `model` | No | `"BAAI/bge-reranker-base"` | Cohere default is `"rerank-v3.5"` |
| `fetch_k` | No | `max(4 * top_k, 20)` | Candidates fetched from the vector store and scored. Raised to `top_k` if set lower. |
| `enabled` | No | `true` | Set `false` to switch reranking off while keeping the rest of the block |

Any other key is passed through to the provider. `cross_encoder` also accepts `batch_size` and `device`.

```yaml
retriever:
  top_k: 5
  rerank:
    provider: "cross_encoder"
    model: "BAAI/bge-reranker-base"
    fetch_k: 25
```

Install the provider you chose: `pip install ragwire[rerank]` for `cross_encoder`, or `pip install ragwire[cohere]` for `cohere`. Neither is pulled in by the base install. A missing package raises `ImportError` at startup rather than on the first query.

!!! warning "Reranking applies to `retrieve()` only"
    `hybrid_search()` and `mmr_search()` are low-level primitives and deliberately ignore the rerank config, so they stay predictable when you compose your own pipeline.

---

### `sources` section

Optional. A list of places `rag.sync()` reconciles the collection against. See [Sync Sources](cookbook/syncing_sources.md).

Every entry needs a `type`. Remaining keys are that type's own settings, plus `extensions` and `name`, which every source accepts.

**`type: local`**

| Key | Required | Default | Description |
|---|---|---|---|
| `path` | Yes | n/a | Directory or single file |
| `recursive` | No | `false` | Descend into subdirectories |
| `extensions` | No | all | Extensions to include, with or without the leading dot |

**`type: s3`** (needs `pip install ragwire[s3]`)

| Key | Required | Default | Description |
|---|---|---|---|
| `bucket` | Yes | n/a | Bucket name |
| `prefix` | No | `""` | Key prefix. Empty means the whole bucket. |
| `cache_dir` | No | `.ragwire_cache` | Where objects are downloaded before ingestion |
| `region` | No | boto3 default | AWS region |
| `endpoint_url` | No | AWS | Point at MinIO, R2, B2 or another S3-compatible store |
| `aws_access_key_id` / `aws_secret_access_key` | No | boto3 chain | Explicit credentials |

```yaml
sources:
  - type: local
    path: "./documents"
    recursive: true
  - type: s3
    bucket: "my-filings"
    prefix: "2026/"
```

---

### `generation` section

Entirely optional. Omit the block and the defaults apply. Controls `rag.query()` only.

| Key | Required | Default | Description |
|---|---|---|---|
| `max_context_chars` | No | `12000` | Total character budget for the source block sent to the LLM |
| `system_prompt` | No | built-in | Override the grounding instructions. Use `{sentinel}` where the model should be told to refuse. |

```yaml
generation:
  max_context_chars: 12000
```

Chunks are added to the context in rank order until the budget runs out. A chunk crossing the limit is truncated; one that would land under 200 characters is dropped instead of being given a source number. Only chunks that fit can be cited.

!!! warning "Raise `llm.num_ctx` alongside `max_context_chars`"
    A default chunk is 10,000 characters, so five of them overflow most context windows. If you raise the budget, raise the model's context window to match. Roughly four characters per token is a safe estimate.

!!! warning "A custom `system_prompt` must keep `{sentinel}`"
    It is replaced with the refusal token. Drop it and refusal detection stops working, so the model will answer from general knowledge whenever your documents come up short.

---

### `ingestion` section

Entirely optional. Omit the block and the defaults below apply, which are
deliberately conservative: single-threaded, no chunk deduplication, and the same
behaviour as earlier versions apart from retries.

| Key | Required | Default | Description |
|---|---|---|---|
| `workers` | No | `1` | Documents prepared in parallel. Loading, splitting and the metadata LLM call run concurrently; writes always stay sequential. |
| `batch_size` | No | `64` | Chunks per write request. Bounds request size on large documents. |
| `retries` | No | `2` | Retries per failed write or metadata call, with exponential backoff. |
| `replace_changed` | No | `true` | When a file's content has changed, delete the old version's chunks before writing the new ones. |
| `dedup_chunks` | No | `false` | Drop chunks whose text repeats within the same document. |

```yaml
ingestion:
  workers: 4              # raise for large batches
  batch_size: 64
  retries: 2
  replace_changed: true
  dedup_chunks: false
```

**`workers`** is the setting worth changing for large batches. Ingest time is
dominated by document conversion and the per-document LLM call, and those are what
run in parallel. Results are always applied in input order, so stats and logs stay
deterministic no matter which document finishes first.

!!! warning "Leave `workers: 1` on local file storage"
    Local Qdrant (`vectorstore.url` pointing at a folder rather than an HTTP URL)
    takes an exclusive lock and is single-process only.

**`retries`** applies to network failures. Programming errors such as `TypeError`
and `AttributeError` are re-raised immediately rather than retried, so a genuine
bug surfaces at once instead of after several backoff delays.

**`dedup_chunks`** helps with long filings that repeat the same boilerplate across
sections, where duplicates otherwise crowd real content out of the top-k. Every
chunk carries a `content_hash` in its metadata whether or not this is enabled.

---

### MarkItDownLoader

Converts documents (PDF, DOCX, XLSX, PPTX, TXT, MD) to markdown text.

```python
from ragwire import MarkItDownLoader
```

**When to use `MarkItDownLoader` directly:** Use it when you need to convert documents to text before passing them to a custom pipeline, or when you want to inspect/transform the text before ingestion.

#### `MarkItDownLoader.load(file_path)`

| Parameter | Type | Required | Description |
|---|---|---|---|
| `file_path` | `str` | Yes | Path to the document |

**Returns:** `dict`

```python
{
    "success": True,
    "text_content": "# Apple Inc.\n\n...",  # Markdown text
    "file_name": "Apple_10k_2025.pdf",
    "file_type": "pdf",
    "error": None                            # Error message if success=False
}
```

```python
loader = MarkItDownLoader()
result = loader.load("data/Apple_10k_2025.pdf")

if result["success"]:
    print(f"Loaded {len(result['text_content'])} characters")
else:
    print(f"Error: {result['error']}")
```

#### `loader.load_batch(file_paths)`

Load multiple documents in one call. Returns results in the same order as the input list.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `file_paths` | `list[str]` | Yes | List of file paths to load |

**Returns:** `list[dict]`, the same structure as `load()` for each file.

```python
loader = MarkItDownLoader()
results = loader.load_batch(["doc1.pdf", "doc2.pdf", "doc3.docx"])

for result in results:
    if result["success"]:
        print(f"{result['file_name']}: {len(result['text_content'])} chars")
    else:
        print(f"{result['file_name']}: {result['error']}")
```

#### `loader.load_directory(directory, extensions, recursive)`

Load all supported documents from a directory.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `directory` | `str` | Yes | n/a | Path to directory |
| `extensions` | `list[str]` | No | all supported | File extensions to include |
| `recursive` | `bool` | No | `False` | Scan subdirectories |

**Returns:** `list[dict]`

```python
loader = MarkItDownLoader()
results = loader.load_directory("data/", extensions=[".pdf", ".docx"], recursive=True)
texts = [r["text_content"] for r in results if r["success"]]
```

---

### Text Splitters

```python
from ragwire import get_splitter, get_markdown_splitter, get_code_splitter
```

All splitters return a `RecursiveCharacterTextSplitter` instance with a `.split_text(text)` method.

**Choosing a splitter:**
- `get_markdown_splitter`: best for PDF/DOCX/reports (converted to markdown by MarkItDown); respects document structure
- `get_splitter`: best for plain text, HTML, or any content without markdown headers
- `get_code_splitter`: best for source code files; splits on class/function boundaries

**Chunk size guidance:** Larger chunks (8k to 12k chars) preserve more context per chunk, which suits long-form financial and legal documents. Smaller chunks (500 to 2k chars) give more precise retrieval, which suits FAQ-style content. `chunk_overlap` prevents context being cut mid-sentence; 20% of chunk size is a sensible default.

#### `get_markdown_splitter(chunk_size, chunk_overlap)`

Splits on markdown headers first (`##`, `###`, `####`), then paragraphs. Best for PDF/DOCX converted via MarkItDown.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `chunk_size` | `int` | `1000` | Max characters per chunk |
| `chunk_overlap` | `int` | `200` | Overlap between chunks |

```python
splitter = get_markdown_splitter(chunk_size=10000, chunk_overlap=2000)
chunks = splitter.split_text(text)
print(f"{len(chunks)} chunks")
```

#### `get_splitter(chunk_size, chunk_overlap, separators)`

Generic recursive splitter. Splits on `\n\n` → `\n` → ` ` → `""`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `chunk_size` | `int` | `1000` | Max characters per chunk |
| `chunk_overlap` | `int` | `200` | Overlap between chunks |
| `separators` | `list[str]` | `["\n\n", "\n", " ", ""]` | Custom separators |

```python
splitter = get_splitter(chunk_size=5000, chunk_overlap=500)
chunks = splitter.split_text(text)
```

#### `get_code_splitter(chunk_size, chunk_overlap)`

Splits on code structure: `class`, `def`, comments. Best for source code files.

```python
splitter = get_code_splitter(chunk_size=2000, chunk_overlap=200)
chunks = splitter.split_text(source_code)
```

---

### get_embedding

Factory function. Returns an embedding model instance for the configured provider.

```python
from ragwire import get_embedding
```

#### `get_embedding(config)`

| Parameter | Type | Required | Description |
|---|---|---|---|
| `config` | `dict` | Yes | Provider config dict with `provider` key |

**Supported providers:** `ollama`, `openai`, `openrouter`, `huggingface`, `google`, `fastembed`

**Returns:** Embedding model with `.embed_query(text)` and `.embed_documents(texts)` methods.

```python
# Ollama
embedding = get_embedding({
    "provider": "ollama",
    "model": "nomic-embed-text",
    "base_url": "http://localhost:11434",
})

# OpenAI
embedding = get_embedding({
    "provider": "openai",
    "model": "text-embedding-3-small",
})

# OpenRouter
embedding = get_embedding({
    "provider": "openrouter",
    "model": "nvidia/llama-nemotron-embed-vl-1b-v2:free",
    "api_key": "${OPENROUTER_API_KEY}",
})

# HuggingFace
embedding = get_embedding({
    "provider": "huggingface",
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "model_kwargs": {"device": "cpu"},
})

vector = embedding.embed_query("What is Apple's revenue?")
print(f"Dimension: {len(vector)}")
```

---

### MetadataExtractor

Extract structured metadata from document text using an LLM.

```python
from ragwire import MetadataExtractor
```

#### `MetadataExtractor(llm, schema_model)`

Uses `with_structured_output` with a Pydantic model for reliable, type-safe extraction, so there is no manual JSON parsing.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `llm` | `Any` | Yes | LangChain chat model instance |
| `schema_model` | `BaseModel` | No | Pydantic model defining fields and types. Defaults to `FinancialMetadata` |

```python
from ragwire import MetadataExtractor, FinancialMetadata
from langchain_ollama import ChatOllama

llm = ChatOllama(model="qwen3.5:9b", base_url="http://localhost:11434")

# Default: uses the FinancialMetadata schema
# (company_name, doc_type, fiscal_quarter, fiscal_year)
extractor = MetadataExtractor(llm)

# Custom Pydantic schema
from pydantic import BaseModel, Field
from typing import Optional, List

class MySchema(BaseModel):
    organization: Optional[str] = Field(None, description="Organization name in lowercase")
    doc_type: Optional[str] = Field(None, description="contract | policy | report")
    effective_year: Optional[int] = Field(None, description="Year the document is effective")
    tags: Optional[List[str]] = Field(None, description="List of topic tags")

extractor = MetadataExtractor(llm, schema_model=MySchema)
```

#### `extractor.extract(text, stored_values)`

| Parameter | Type | Required | Description |
|---|---|---|---|
| `text` | `str` | Yes | Document text (first 10,000 chars used) |
| `stored_values` | `dict` | No | Existing field values from the collection. When provided, the LLM reuses stored names (e.g. `"apple inc."`) instead of extracting inconsistent variants. Pass `rag.get_field_values(fields)` or use `rag.extract_metadata()` which injects this automatically. |

**Returns:** `dict`

```python
{
    "company_name": "apple inc.",
    "doc_type": "10-k",
    "fiscal_quarter": None,
    "fiscal_year": 2025
}
```

```python
# Basic extraction
metadata = extractor.extract(document_text)

# With grounding, the LLM reuses stored entity names
stored = rag.get_field_values(rag.filter_fields)
metadata = extractor.extract(document_text, stored_values=stored)
print(metadata)
```

#### `MetadataExtractor.from_yaml(llm, yaml_path)`

Create an extractor from a YAML file. Builds a Pydantic model dynamically from the field definitions.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `llm` | `Any` | Yes | LangChain chat model instance |
| `yaml_path` | `str` | Yes | Path to metadata YAML config file |

**Returns:** `MetadataExtractor`

```python
extractor = MetadataExtractor.from_yaml(llm, "metadata.yaml")
metadata = extractor.extract(document_text)
```

See [Custom Metadata](custom_metadata.md) for the YAML format including `type` and `values` field options.

---

#### `extractor.extract_batch(texts)`

| Parameter | Type | Description |
|---|---|---|
| `texts` | `list[str]` | List of document texts |

**Returns:** `list[dict]`

---

### DocumentMetadata

Pydantic schema for chunk metadata. Useful for type-checking or building typed wrappers.

```python
from ragwire import DocumentMetadata
```

```python
meta = DocumentMetadata(
    company_name="apple",
    doc_type="10-k",
    fiscal_year=[2025],
    source="/data/Apple_10k_2025.pdf",
    file_name="Apple_10k_2025.pdf",
    file_type="pdf",
    file_hash="abc123...",
    chunk_id="abc123_0",
    chunk_hash="def456...",
    chunk_index=0,
    total_chunks=42,
)
print(meta.model_dump())
```

See [Metadata & Filtering](metadata.md) for the full field reference.

---

### Logging

```python
from ragwire import setup_logging, setup_colored_logging
```

Use `setup_logging` for plain text logs (production, log files). Use `setup_colored_logging` during development, since it color-codes log levels so warnings and errors stand out at a glance.

#### `setup_logging(log_level, log_file, console_output, format_string)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `log_level` | `str` | `"INFO"` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `log_file` | `str` | `None` | Optional path to write logs to file |
| `console_output` | `bool` | `True` | Print logs to stdout |
| `format_string` | `str` | `None` | Custom log format string |

**Returns:** `logging.Logger`

```python
logger = setup_logging(log_level="DEBUG", log_file="logs/rag.log")
logger.info("Pipeline started")
```

#### `setup_colored_logging(log_level, log_file)`

Same as `setup_logging` but with colored console output: errors in red, warnings in yellow, info in green. Useful during development to spot issues quickly.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `log_level` | `str` | `"INFO"` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `log_file` | `str` | `None` | Optional path to write plain-text logs to file |

**Returns:** `logging.Logger`

```python
from ragwire import setup_colored_logging

logger = setup_colored_logging(log_level="DEBUG")
logger.info("Pipeline started")   # green
logger.warning("Slow response")   # yellow
logger.error("LLM call failed")   # red
```

You can also enable colored logging from `config.yaml`, with no code change:

```yaml
logging:
  level: "INFO"
  colored: true
  console_output: true
  # log_file: "logs/rag.log"   # uncomment to also write to file
```

---

## Command Line

Installing the package puts a `ragwire` command on your PATH.

| Command | What it does |
|---|---|
| `ragwire --version` | Print the installed version |
| `ragwire mcp serve [--config CONFIG] [--name NAME]` | Run the MCP server over stdio. Requires `pip install ragwire[mcp]`. See [MCP Server](cookbook/mcp_server.md). |
| `ragwire ingest PATH [--config CONFIG] [--recursive]` | Ingest a file or directory |
| `ragwire sync [--config CONFIG] [--dry-run] [--no-delete]` | Reconcile the collection against its configured sources |
| `ragwire eval GOLDEN [--config CONFIG] [--top-k K] [--compare-rerank]` | Score retrieval against a golden set |

`--config` defaults to `config.yaml` for every subcommand.

```bash
ragwire ingest ./documents --recursive
ragwire eval golden.yaml --compare-rerank
```

`ragwire ingest` exits non-zero when every file failed, so it can be used in a script without checking its output.

---

## MCP Tools

`ragwire.mcp` holds the tool implementations behind the MCP server. They are plain functions with no MCP dependency, so they work anywhere an agent needs to reach a collection.

```python
from ragwire.mcp import search_documents, answer_question, get_filter_context, collection_stats
```

| Function | Signature | Returns |
|---|---|---|
| `search_documents` | `(rag, query, top_k=5, filters=None)` | Matching passages with sources, as text |
| `answer_question` | `(rag, question, top_k=5, filters=None)` | A cited answer, or a refusal telling the agent not to fill the gap itself |
| `get_filter_context` | `(rag, query="")` | The fields and stored values available for filtering |
| `collection_stats` | `(rag)` | Collection name, chunk count, vector size |

`filters` accepts a dict or a JSON string, since models frequently send a string where a schema asked for an object. An unparseable value raises `ValueError` with a message the agent can act on.

Chunk metadata is filtered before it reaches the agent: `content_hash`, `total_chunks`, `chunk_index`, `metadata_status` and any key starting with `_` are hidden, since they describe how a chunk was stored rather than what it says.

#### `build_server(rag, name="ragwire")`

Build a `FastMCP` server exposing the four tools above. Raises `ImportError` when the `mcp` package is missing.

```python
from ragwire import RAGWire
from ragwire.mcp import build_server

build_server(RAGWire("config.yaml")).run()
```

#### `serve(config_path="config.yaml", name="ragwire")`

Build and run the server over stdio. This is what `ragwire mcp serve` calls.

---

## Evaluation API

`ragwire.eval` scores retrieval against a golden set. No extra install is needed. See [Measure Retrieval Quality](cookbook/evaluation.md) for the guide.

```python
from ragwire.eval import GoldenSet, evaluate, sweep
```

---

### GoldenSet

A list of queries paired with the documents that should be retrieved for them.

#### `GoldenSet.from_file(path)`

Load a golden set from YAML or JSON.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `path` | `str` \| `Path` | Yes | n/a | Path to a `.yaml`, `.yml` or `.json` file |

**Returns:** `GoldenSet`

**Raises:** `FileNotFoundError` if the file is missing, `ValueError` if an entry has no `expected` key or an empty query.

The file is either a bare list of entries, or a mapping with a `queries` key plus `match_field` and `match_mode`:

```yaml
- query: "What was Apple's net income in fiscal 2025?"
  expected: ["apple_10k_2025.pdf"]
  filters: {company_name: "apple"}   # optional, passed to retrieve()
  note: "vague phrasing, fails first when chunk_size grows"  # optional
```

| Entry key | Type | Required | Description |
|---|---|---|---|
| `query` | `str` | Yes | The search query |
| `expected` | `str` \| `list[str]` | Yes | Identifiers that count as a correct hit. A single string is accepted unwrapped. |
| `filters` | `dict` | No | Metadata filters passed to `retrieve()` for this query |
| `note` | `str` | No | Free text. Ignored by scoring. |

| Set-level key | Default | Description |
|---|---|---|
| `match_field` | `"source"` | Metadata field compared against `expected` |
| `match_mode` | `"basename"` | `"basename"` \| `"exact"` \| `"contains"` |

#### `GoldenSet.from_data(data)`

Same as `from_file` but takes already-parsed data. Useful in tests.

---

### evaluate

#### `evaluate(rag, golden, top_k, label, **retrieve_kwargs)`

Run every golden query and score what comes back.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `rag` | `RAGWire` | Yes | n/a | The pipeline to evaluate |
| `golden` | `GoldenSet` | Yes | n/a | Queries to run |
| `top_k` | `int` | No | `5` | How many documents to retrieve and score at |
| `label` | `str` | No | `"default"` | Name for this run, shown in output tables |
| `**retrieve_kwargs` | | No | nothing | Passed to `retrieve()`, so `rerank=False` works here |

**Returns:** `EvalResult`

| Attribute | Type | Description |
|---|---|---|
| `metrics` | `dict[str, float]` | Averaged `recall`, `mrr`, `hit_rate`, `precision` |
| `per_query` | `list[QueryResult]` | One entry per golden query |
| `failures` | `list[QueryResult]` | Queries that retrieved nothing correct |
| `to_table()` | `str` | Formatted summary, also what `print()` shows |

Each `QueryResult` carries `query`, `expected`, `retrieved` (rank-ordered, with `"<miss>"` for documents that matched nothing), `metrics` and `missed`.

A query that raises during retrieval is logged and scored as zero rather than aborting the run.

---

### sweep

#### `sweep(rag, golden, variants, top_k)`

Evaluate several retrieval settings against the same golden set.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `rag` | `RAGWire` | Yes | n/a | The pipeline to evaluate |
| `golden` | `GoldenSet` | Yes | n/a | Queries to run |
| `variants` | `dict[str, dict]` | Yes | n/a | Label to `retrieve()` kwargs. The first entry is the baseline later rows are compared against. |
| `top_k` | `int` | No | `5` | Default cutoff, overridable per variant |

**Returns:** `SweepResult`, with `.results`, `.best` and `.to_table()`.

```python
print(sweep(rag, golden, {
    "no rerank": {"rerank": False},
    "reranked":  {"rerank": True},
}))
```

---

### Metric functions

Exported for building custom reports. Each takes `retrieved` in rank order and `expected` as the correct set, plus an optional `k` cutoff.

| Function | Returns |
|---|---|
| `recall_at_k(retrieved, expected, k)` | Fraction of expected documents found |
| `precision_at_k(retrieved, expected, k)` | Fraction of results that were correct |
| `hit_rate_at_k(retrieved, expected, k)` | `1.0` if anything correct was found |
| `reciprocal_rank(retrieved, expected, k)` | `1 / rank` of the first correct result |
| `score_query(retrieved, expected, k)` | All four as a dict |
| `mean_metrics(per_query)` | Averages a list of metric dicts |

---

## Low-level / Advanced API

These APIs are exported for advanced use cases such as custom pipelines, direct vector store access, or building on top of RAGWire internals. Most users will not need these directly.

---

### QdrantStore

Direct Qdrant collection management. Use this when you need fine-grained control over the vector store outside of `RAGWire`.

```python
from ragwire import QdrantStore
```

#### `QdrantStore(config, embedding, collection_name)`

| Parameter | Type | Required | Description |
|---|---|---|---|
| `config` | `dict` | Yes | Vectorstore config (`url`, `api_key`) |
| `embedding` | `Any` | Yes | Embedding model instance |
| `collection_name` | `str` | No | Collection name |

#### Methods

| Method | Returns | Description |
|---|---|---|
| `set_collection(name)` | `None` | Set active collection |
| `get_store(use_sparse)` | `QdrantVectorStore` | Get LangChain vectorstore instance |
| `create_collection(use_sparse)` | `None` | Create a new collection |
| `delete_collection()` | `None` | Delete the collection |
| `collection_exists()` | `bool` | Check if collection exists |
| `file_hash_exists(file_hash)` | `bool` | Check if file already ingested |
| `get_collection_info()` | `CollectionInfo` | Get Qdrant collection metadata |
| `get_metadata_keys()` | `list[str]` | Scroll one point, return all metadata field names |
| `get_field_values(fields, limit)` | `dict` | Unique values per field via Qdrant facet API |
| `create_payload_indexes(fields)` | `None` | Create keyword indexes for facet API (auto-called during ingestion) |

```python
store = QdrantStore(
    config={"url": "http://localhost:6333"},
    embedding=embedding,
    collection_name="my_docs",
)
store.create_collection(use_sparse=True)
vectorstore = store.get_store(use_sparse=True)

docs = vectorstore.similarity_search("revenue", k=5)
```

#### `store.get_metadata_keys()`

Scrolls one point from the collection and returns all metadata field names present. Use this when you don't know what fields were stored, for example when inspecting a collection built by someone else, or verifying custom metadata was extracted correctly.

```python
fields = store.get_metadata_keys()
# → ['company_name', 'doc_type', 'fiscal_year', 'file_name', 'chunk_index', ...]
```

#### `store.get_field_values(fields, limit)`

Returns unique values for each requested field using Qdrant's facet API. Requires payload indexes on those fields, so call `create_payload_indexes()` first if you haven't ingested via `RAGWire` (which does this automatically).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `fields` | `list[str]` | n/a | Field names (without `metadata.` prefix) |
| `limit` | `int` | `50` | Max unique values per field |

**Returns:** `dict[str, list]`

```python
# Discover fields first, then get values for the ones you care about
fields = store.get_metadata_keys()
# → ['company_name', 'doc_type', 'fiscal_year', ...]

values = store.get_field_values(["company_name", "doc_type"])
# → {'company_name': ['apple', 'microsoft'], 'doc_type': ['10-k', '10-q']}

# High-cardinality field, so raise the limit
values = store.get_field_values(["file_name"], limit=200)
```

!!! note "Using `RAGWire` instead?"
    If you're using `RAGWire`, prefer `rag.filter_fields` + `rag.get_field_values()` for filter prompts, and `rag.discover_metadata_fields()` for collection inspection. They are thin wrappers over these same methods and don't require you to manage the `QdrantStore` instance directly.

---

### Retrieval Functions

Use these when building a custom retrieval layer outside of `RAGWire`.

```python
from ragwire import get_retriever, hybrid_search, mmr_search
```

**Choosing a search strategy:**

| Strategy | Use when |
|---|---|
| `similarity` | General semantic search; fast, good default |
| `hybrid` | Queries mix semantic meaning with exact keywords (e.g. ticker symbols, product names, IDs) |
| `mmr` | You want diverse results, avoiding 5 nearly identical chunks from the same page |

#### `get_retriever(vectorstore, top_k, search_type)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `vectorstore` | `QdrantVectorStore` | n/a | Vector store instance |
| `top_k` | `int` | `5` | Number of results |
| `search_type` | `str` | `"similarity"` | `"similarity"`, `"mmr"`, `"hybrid"` |

**Returns:** LangChain retriever with `.invoke(query)` method.

#### `hybrid_search(vectorstore, query, k, filters)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `vectorstore` | `QdrantVectorStore` | n/a | Vector store instance |
| `query` | `str` | n/a | Search query |
| `k` | `int` | `5` | Number of results |
| `filters` | `dict` | `None` | Plain metadata filter dict (same format as `rag.retrieve()` filters) |

**Returns:** `list[Document]`

#### `mmr_search(vectorstore, query, k, fetch_k, lambda_mult, filters)`

Maximal Marginal Relevance retrieves diverse, non-redundant results. Use this when a regular similarity search returns several near-identical chunks from the same section of a document, and you want results spread across different parts.

`fetch_k` controls how many candidates are retrieved first, then MMR selects the most diverse `k` from them. A larger `fetch_k` gives MMR more candidates to choose from. `lambda_mult` controls the balance: `0.0` = maximise diversity, `1.0` = maximise relevance (same as similarity search), `0.5` = balanced default.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `vectorstore` | `QdrantVectorStore` | n/a | Vector store instance |
| `query` | `str` | n/a | Search query |
| `k` | `int` | `5` | Number of results to return |
| `fetch_k` | `int` | `20` | Candidates fetched before MMR selection |
| `lambda_mult` | `float` | `0.5` | Diversity (`0.0` = max diverse, `1.0` = max relevant) |
| `filters` | `dict` | `None` | Plain metadata filter dict (same format as `rag.retrieve()` filters) |

**Returns:** `list[Document]`

```python
# Balanced, a good default
results = mmr_search(vectorstore, "Apple revenue and earnings", k=5)

# More diverse, useful when documents are long and repetitive
results = mmr_search(vectorstore, "Apple revenue and earnings", k=5, lambda_mult=0.3)
```

---

### Hashing Utilities

Used internally by the pipeline for SHA256 deduplication. Exposed for custom ingestion workflows.

**Why deduplication matters:** Without it, re-running ingestion on the same files doubles the chunks in Qdrant, degrading retrieval quality and wasting storage. RAGWire checks `file_hash` before ingesting: if a file with the same hash already exists in the collection, the file is skipped entirely.

```python
from ragwire import sha256_text, sha256_file_from_path, sha256_chunk
```

| Function | Parameters | Returns | Description |
|---|---|---|---|
| `sha256_text(text)` | `text: str` | `str` | SHA256 of a text string |
| `sha256_file_from_path(path)` | `path: str \| Path` | `str` | SHA256 of a file (streamed, memory-efficient) |
| `sha256_chunk(chunk_id, content)` | `chunk_id: str, content: str` | `str` | SHA256 of a chunk (id + content combined) |

```python
from ragwire import sha256_file_from_path

file_hash = sha256_file_from_path("data/Apple_10k_2025.pdf")
print(file_hash)  # "a1b2c3d4..."
```

---

### get_logger

Get a child logger under the `ragwire` namespace. Used internally by all modules.

```python
from ragwire import get_logger

logger = get_logger(__name__)
logger.info("Custom module log")
```
