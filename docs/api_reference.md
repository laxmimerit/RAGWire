# API Reference

All public APIs are importable directly from `ragwire`:

```python
from ragwire import RAGPipeline, MarkItDownLoader, get_embedding, QdrantStore, ...
```

---

## Core API

These are the primary user-facing APIs. Most applications only need these.

---

### RAGPipeline

The main orchestrator. Handles the full pipeline from config loading to ingestion and retrieval.

```python
from ragwire import RAGPipeline
```

#### `RAGPipeline(config_path)`

Initialize the pipeline from a YAML config file.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `config_path` | `str` | Yes | Path to `config.yaml` |

**Raises:**

- `FileNotFoundError` — config file not found
- `ValueError` — missing required config keys (e.g. `llm.model`)

```python
pipeline = RAGPipeline("config.yaml")
```

---

#### `pipeline.ingest_documents(file_paths)`

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
stats = pipeline.ingest_documents([
    "data/Apple_10k_2025.pdf",
    "data/Microsoft_10k_2025.pdf",
])
print(f"Processed: {stats['processed']}, Chunks: {stats['chunks_created']}")
```

---

#### `pipeline.retrieve(query, top_k, filters)`

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

```python
# Basic retrieval
results = pipeline.retrieve("What is the total revenue?", top_k=5)

# With filters
results = pipeline.retrieve(
    "What is the net income?",
    top_k=5,
    filters={"company_name": "apple", "fiscal_year": 2025}
)

for doc in results:
    print(doc.metadata.get("company_name"))
    print(doc.page_content[:300])
```

---

#### `pipeline.hybrid_search(query, k, filters)`

Perform hybrid search combining dense (semantic) and sparse (keyword) vectors. Requires `use_sparse: true` in config.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `query` | `str` | Yes | — | Search query |
| `k` | `int` | No | `5` | Number of results |
| `filters` | `dict` | No | `None` | Metadata filters |

**Returns:** `list[Document]`

```python
results = pipeline.hybrid_search(
    "Apple revenue fiscal 2025",
    k=5,
    filters={"company_name": "apple"}
)
```

---

#### `pipeline.get_stats()`

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
stats = pipeline.get_stats()
print(f"Collection: {stats['collection_name']}, Chunks: {stats['total_documents']}")
```

---

### MarkItDownLoader

Converts documents (PDF, DOCX, XLSX, PPTX, TXT, MD) to markdown text.

```python
from ragwire import MarkItDownLoader
```

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

---

### Text Splitters

```python
from ragwire import get_splitter, get_markdown_splitter, get_code_splitter
```

All splitters return a `RecursiveCharacterTextSplitter` instance with a `.split_text(text)` method.

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

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
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

See [Custom Metadata via YAML](metadata.md#custom-metadata-via-yaml-file) for the YAML format.

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

### setup_logging

Configure logging for the pipeline.

```python
from ragwire import setup_logging
```

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

---

## Low-level / Advanced API

These APIs are exported for advanced use cases — custom pipelines, direct vector store access, or building on top of RAGWire internals. Most users will not need these directly.

---

### QdrantStore

Direct Qdrant collection management. Use this when you need fine-grained control over the vector store outside of `RAGPipeline`.

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

---

### Retrieval Functions

Use these when building a custom retrieval layer outside of `RAGPipeline`.

```python
from ragwire import get_retriever, hybrid_search, mmr_search
```

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

Maximal Marginal Relevance — retrieves diverse, non-redundant results.

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
results = mmr_search(
    vectorstore,
    "Apple revenue and earnings",
    k=5,
    fetch_k=20,
    lambda_mult=0.7,
)
```

---

### Hashing Utilities

Used internally by the pipeline for SHA256 deduplication. Exposed for custom ingestion workflows.

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
