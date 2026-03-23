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

- `FileNotFoundError` — config file not found
- `ValueError` — missing required config keys (e.g. `llm.model`)

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
    "errors": []          # List of {"file": ..., "error": ...} dicts
}
```

```python
stats = rag.ingest_documents([
    "data/Apple_10k_2025.pdf",
    "data/Microsoft_10k_2025.pdf",
])
print(f"Processed: {stats['processed']}, Chunks: {stats['chunks_created']}")
```

A progress bar (`tqdm`) is shown automatically while ingestion runs.

---

#### `rag.ingest_directory(directory, recursive, extensions)`

Ingest all supported documents from a directory. Internally calls `ingest_documents()`.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `directory` | `str` | Yes | — | Path to the directory |
| `recursive` | `bool` | No | `False` | Search subdirectories |
| `extensions` | `list[str]` | No | loader config | File extensions to include |

**Returns:** `dict` — same stats dict as `ingest_documents()`

```python
# Ingest all PDFs/DOCX in a folder
stats = rag.ingest_directory("data/")

# Recursively include subdirectories
stats = rag.ingest_directory("data/", recursive=True)

# Only specific extensions
stats = rag.ingest_directory("data/", extensions=[".pdf"])
```

---

#### `rag.retrieve(query, top_k, filters)`

Retrieve the most relevant chunks for a query.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `query` | `str` | Yes | — | Search query |
| `top_k` | `int` | No | config value | Number of results to return |
| `filters` | `dict` | No | `None` | Metadata filters (see [Metadata & Filtering](metadata.md)) |

**Returns:** `list[Document]`

Each `Document` has:

- `doc.page_content` — the chunk text
- `doc.metadata` — dict with all metadata fields (see [Metadata Schema](metadata.md#metadata-schema))

**Filter behaviour:**

- If `filters` is passed → used as-is, no LLM call
- If `filters` is not passed → LLM automatically extracts filters from the query

**When to use auto-filter vs explicit filters:** Use explicit filters in programmatic pipelines where you control the inputs (faster, zero LLM overhead). Use auto-filter in user-facing chatbots where the user types natural language queries and you want the system to figure out the right filters automatically.

```python
# Explicit filters — LLM extraction skipped
results = rag.retrieve(
    "What is the net income?",
    top_k=5,
    filters={"company_name": "apple", "fiscal_year": 2025}
)

# No filters passed — LLM extracts {"company_name": "apple", "fiscal_year": 2025} from the query
results = rag.retrieve("What is Apple's net income for 2025?")

for doc in results:
    print(doc.metadata.get("company_name"))
    print(doc.page_content[:300])
```

---

#### `rag.hybrid_search(query, k, filters)`

Perform hybrid search combining dense (semantic) and sparse (keyword) vectors. Requires `use_sparse: true` in config.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `query` | `str` | Yes | — | Search query |
| `k` | `int` | No | `5` | Number of results |
| `filters` | `dict` | No | `None` | Metadata filters |

**Returns:** `list[Document]`

```python
results = rag.hybrid_search(
    "Apple revenue fiscal 2025",
    k=5,
    filters={"company_name": "apple"}
)
```

---

#### `rag.discover_metadata_fields()`

Return all metadata field names present in the collection. Scrolls one point — fast regardless of collection size.

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
| `fields` | `str \| list[str]` | Yes | — | Field name or list of field names |
| `limit` | `int` | No | `50` | Max unique values to return per field. Increase for high-cardinality fields (e.g. `file_name`). |

**Returns:**
- `list` — if `fields` is a `str`
- `dict[str, list]` — if `fields` is a `list`

```python
# Single field — returns a list of up to 50 unique values
rag.get_field_values("company_name")
# → ['apple', 'microsoft', 'google']

# Multiple fields — returns a dict
rag.get_field_values(["company_name", "doc_type"])
# → {'company_name': ['apple', 'microsoft', 'google'], 'doc_type': ['10-k', '10-q']}

# High-cardinality field — raise the limit
rag.get_field_values("file_name", limit=200)
# → ['Apple_10k_2025.pdf', 'Microsoft_10k_2025.pdf', ...]

# Typical agent workflow
fields = rag.discover_metadata_fields()
values = rag.get_field_values(["company_name", "doc_type"])
results = rag.retrieve("revenue", filters={"company_name": values["company_name"][0]})
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

**Returns:** `list[dict]` — same structure as `load()` for each file.

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
| `directory` | `str` | Yes | — | Path to directory |
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
- `get_markdown_splitter` — best for PDF/DOCX/reports (converted to markdown by MarkItDown); respects document structure
- `get_splitter` — best for plain text, HTML, or any content without markdown headers
- `get_code_splitter` — best for source code files; splits on class/function boundaries

**Chunk size guidance:** Larger chunks (8k–12k chars) preserve more context per chunk — good for long-form financial/legal docs. Smaller chunks (500–2k chars) give more precise retrieval — good for FAQ-style content. `chunk_overlap` prevents context being cut mid-sentence; 20% of chunk size is a sensible default.

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

Factory function — returns an embedding model instance for the configured provider.

```python
from ragwire import get_embedding
```

#### `get_embedding(config)`

| Parameter | Type | Required | Description |
|---|---|---|---|
| `config` | `dict` | Yes | Provider config dict with `provider` key |

**Supported providers:** `ollama`, `openai`, `huggingface`, `google`, `fastembed`

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

#### `MetadataExtractor(llm, prompt_template)`

| Parameter | Type | Required | Description |
|---|---|---|---|
| `llm` | `Any` | Yes | LangChain chat model instance |
| `prompt_template` | `str` | No | Custom prompt (uses financial default if not set) |

#### `extractor.extract(text)`

| Parameter | Type | Required | Description |
|---|---|---|---|
| `text` | `str` | Yes | Document text (first 10,000 chars used) |

**Returns:** `dict`

```python
{
    "company_name": "apple",
    "doc_type": "10-k",
    "fiscal_quarter": None,
    "fiscal_year": [2025]
}
```

```python
from langchain_openai import ChatOpenAI
from ragwire import MetadataExtractor

llm = ChatOpenAI(model="gpt-5.4-nano")
extractor = MetadataExtractor(llm)
metadata = extractor.extract(document_text)
print(metadata)
```

#### `MetadataExtractor.from_yaml(llm, yaml_path)`

Create an extractor configured from a YAML file. The YAML defines fields (auto-builds the prompt) or a full `prompt_template`.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `llm` | `Any` | Yes | LangChain chat model instance |
| `yaml_path` | `str` | Yes | Path to metadata YAML config file |

**Returns:** `MetadataExtractor`

```python
extractor = MetadataExtractor.from_yaml(llm, "metadata.yaml")
metadata = extractor.extract(document_text)
```

See [Custom Metadata via YAML](custom_metadata.md#custom-metadata-via-yaml-file) for the YAML format.

---

#### `MetadataExtractor.build_prompt_from_fields(fields)`

Build a JSON extraction prompt from a list of field definitions. Called automatically by `from_yaml()`.

| Parameter | Type | Description |
|---|---|---|
| `fields` | `list[dict]` | Each dict: `name` (str), `description` (str), `values` (list, optional) |

**Returns:** `str` — prompt template with `{content}` placeholder.

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

Use `setup_logging` for plain text logs (production, log files). Use `setup_colored_logging` during development — color-codes log levels so warnings and errors stand out at a glance.

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

Same as `setup_logging` but with colored console output — errors in red, warnings in yellow, info in green. Useful during development to spot issues quickly.

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

You can also enable colored logging from `config.yaml` — no code change needed:

```yaml
logging:
  level: "INFO"
  colored: true
  console_output: true
  # log_file: "logs/rag.log"   # uncomment to also write to file
```

---

## Low-level / Advanced API

These APIs are exported for advanced use cases — custom pipelines, direct vector store access, or building on top of RAGWire internals. Most users will not need these directly.

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

Scrolls one point from the collection and returns all metadata field names present. Use this when you don't know what fields were stored — e.g. inspecting a collection built by someone else, or verifying custom metadata was extracted correctly.

```python
fields = store.get_metadata_keys()
# → ['company_name', 'doc_type', 'fiscal_year', 'file_name', 'chunk_index', ...]
```

#### `store.get_field_values(fields, limit)`

Returns unique values for each requested field using Qdrant's facet API. Requires payload indexes on those fields — call `create_payload_indexes()` first if you haven't ingested via `RAGWire` (which does this automatically).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `fields` | `list[str]` | — | Field names (without `metadata.` prefix) |
| `limit` | `int` | `50` | Max unique values per field |

**Returns:** `dict[str, list]`

```python
# Discover fields first, then get values for the ones you care about
fields = store.get_metadata_keys()
# → ['company_name', 'doc_type', 'fiscal_year', ...]

values = store.get_field_values(["company_name", "doc_type"])
# → {'company_name': ['apple', 'microsoft'], 'doc_type': ['10-k', '10-q']}

# High-cardinality field — raise the limit
values = store.get_field_values(["file_name"], limit=200)
```

!!! note "Using `RAGWire` instead?"
    If you're using `RAGWire`, prefer `rag.discover_metadata_fields()` and `rag.get_field_values()` — they are thin wrappers over these same methods and don't require you to manage the `QdrantStore` instance directly.

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
| `mmr` | You want diverse results — avoids returning 5 nearly identical chunks from the same page |

#### `get_retriever(vectorstore, top_k, search_type)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `vectorstore` | `QdrantVectorStore` | — | Vector store instance |
| `top_k` | `int` | `5` | Number of results |
| `search_type` | `str` | `"similarity"` | `"similarity"`, `"mmr"`, `"hybrid"` |

**Returns:** LangChain retriever with `.invoke(query)` method.

#### `hybrid_search(vectorstore, query, k, filters)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `vectorstore` | `QdrantVectorStore` | — | Vector store instance |
| `query` | `str` | — | Search query |
| `k` | `int` | `5` | Number of results |
| `filters` | `dict` | `None` | Qdrant `Filter` object |

**Returns:** `list[Document]`

#### `mmr_search(vectorstore, query, k, fetch_k, lambda_mult, filters)`

Maximal Marginal Relevance — retrieves diverse, non-redundant results. Use this when a regular similarity search returns several near-identical chunks from the same section of a document, and you want results spread across different parts.

`fetch_k` controls how many candidates are retrieved first, then MMR selects the most diverse `k` from them. A larger `fetch_k` gives MMR more candidates to choose from. `lambda_mult` controls the balance: `0.0` = maximise diversity, `1.0` = maximise relevance (same as similarity search), `0.5` = balanced default.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `vectorstore` | `QdrantVectorStore` | — | Vector store instance |
| `query` | `str` | — | Search query |
| `k` | `int` | `5` | Number of results to return |
| `fetch_k` | `int` | `20` | Candidates fetched before MMR selection |
| `lambda_mult` | `float` | `0.5` | Diversity (`0.0` = max diverse, `1.0` = max relevant) |
| `filters` | `dict` | `None` | Qdrant `Filter` object |

**Returns:** `list[Document]`

```python
# Balanced — good default
results = mmr_search(vectorstore, "Apple revenue and earnings", k=5)

# More diverse — useful when documents are long and repetitive
results = mmr_search(vectorstore, "Apple revenue and earnings", k=5, lambda_mult=0.3)
```

---

### Hashing Utilities

Used internally by the pipeline for SHA256 deduplication. Exposed for custom ingestion workflows.

**Why deduplication matters:** Without it, re-running ingestion on the same files doubles the chunks in Qdrant, degrading retrieval quality and wasting storage. RAGWire checks `file_hash` before ingesting — if a file with the same hash already exists in the collection, the file is skipped entirely.

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
