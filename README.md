<p align="center">
  <img src="https://raw.githubusercontent.com/laxmimerit/RAGWire/main/assets/ragwire.png" alt="RAGWire logo" width="120"/>
</p>

<h1 align="center">RAGWire</h1>
<p align="center">
  Build production-ready RAG pipelines with document ingestion, metadata-aware retrieval, hybrid search, and Qdrant storage in a few lines of Python.
</p>

<p align="center">
  <a href="https://pypi.org/project/ragwire"><img src="https://img.shields.io/pypi/v/ragwire" alt="PyPI"/></a>
  <a href="https://pypi.org/project/ragwire"><img src="https://img.shields.io/pypi/pyversions/ragwire" alt="Python Versions"/></a>
  <a href="https://pepy.tech/project/ragwire"><img src="https://static.pepy.tech/badge/ragwire" alt="Downloads"/></a>
  <a href="https://github.com/laxmimerit/ragwire/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"/></a>
  <a href="https://laxmimerit.github.io/RAGWire/"><img src="https://img.shields.io/badge/docs-live-blue" alt="Documentation"/></a>
  <a href="https://youtube.com/kgptalkie"><img src="https://img.shields.io/badge/YouTube-KGP%20Talkie-red" alt="YouTube"/></a>
</p>

<p align="center">
  <a href="https://laxmimerit.github.io/RAGWire/">
    <img src="https://img.shields.io/badge/📖%20Full%20Documentation-laxmimerit.github.io%2FRAGWire-blue?style=for-the-badge&logo=readthedocs&logoColor=white" alt="Documentation"/>
  </a>
</p>

<p align="center">
  <a href="https://raw.githubusercontent.com/laxmimerit/RAGWire/main/assets/ragwire_explainer_10s.mp4">
    <img src="https://raw.githubusercontent.com/laxmimerit/RAGWire/main/assets/ragwire_explainer_10s.gif" alt="RAGWire 10-second explainer" width="100%"/>
  </a>
</p>

---

## Table of Contents

- [Why RAGWire?](#why-ragwire)
- [Try It In 60 Seconds](#try-it-in-60-seconds)
- [Best For](#best-for)
- [Supported Stack](#supported-stack)
- [Project Status](#project-status)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Embedding Providers](#embedding-providers)
- [Examples](#examples)
- [Production Notes](#production-notes)
- [Security & Privacy](#security--privacy)
- [How RAGWire Fits](#how-ragwire-fits)
- [Component Usage](#component-usage)
- [Package Structure](#package-structure)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## Why RAGWire?

RAGWire gives you the building blocks for document-heavy RAG systems without forcing you into a full application framework. It keeps ingestion, metadata extraction, vector storage, and retrieval configurable, inspectable, and easy to wire into your own apps or agents.

- Ingest PDFs, DOCX, XLSX, PPTX, Markdown, and text files
- Extract structured metadata with your chosen LLM
- Store dense and optional sparse vectors in Qdrant
- Retrieve with metadata filters, MMR, hybrid search, or cross-encoder reranking
- Answer questions with citations, and measure whether retrieval is any good
- Swap LLM and embedding providers through YAML config
- Re-run ingestion safely with SHA256 deduplication

## Try It In 60 Seconds

```bash
pip install ragwire
```

Start Qdrant locally:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

```python
from ragwire import RAGWire

rag = RAGWire("config.yaml")

rag.ingest_directory("data/", recursive=True)

answer = rag.query("What was Apple's revenue in 2025?")
print(answer.formatted())
```

Example output:

```text
Apple reported total net sales of $416.2 billion for fiscal 2025 [1], an
increase of 8% year over year [1].

Sources:
  [1] Apple_10k_2025.pdf
```

If the documents do not answer the question, `query()` says so rather than falling back on the model's own knowledge. To work with the underlying chunks instead:

```python
results = rag.retrieve("What was Apple's revenue in 2025?", top_k=5)
for doc in results:
    print(doc.page_content[:300])
    print(doc.metadata)
```

```text
Apple reported total net sales of ...
{'company_name': 'apple inc.', 'doc_type': '10-k', 'fiscal_year': 2025, 'file_name': 'Apple_10k_2025.pdf'}
```

## Best For

- Financial document QA and SEC filing analysis
- Internal knowledge bases over PDFs and office documents
- Research document search with metadata filters
- Agentic RAG systems that need filter context
- Giving Claude Desktop, Claude Code or Cursor access to your own documents over MCP
- Scheduled pipelines that must stay in step with a folder or an S3 bucket
- Hybrid retrieval applications using Qdrant

## Supported Stack

| Layer | Supported |
|---|---|
| Document loading | PDF, DOCX, XLSX, PPTX, TXT, MD via MarkItDown |
| Vector database | Qdrant |
| Embeddings | Ollama, OpenAI, OpenRouter, HuggingFace, Google, FastEmbed |
| LLM metadata extraction | Ollama, OpenAI, OpenRouter, Gemini, Groq, Anthropic |
| Retrieval | Similarity, MMR, hybrid dense+sparse search, cross-encoder reranking |
| Reranking | Local cross-encoder (sentence-transformers), Cohere Rerank |
| Answer generation | Cited answers with refusal, sync and async |
| Sources | Local folders, S3 and S3-compatible stores, custom connectors |
| Agent integration | MCP server for Claude Desktop, Claude Code, Cursor |
| Evaluation | recall@k, MRR, hit rate, precision, config sweeps |
| Configuration | YAML + environment variables |

## Project Status

RAGWire is in beta and designed for developers building production-style RAG systems. The core ingestion, metadata extraction, Qdrant storage, and retrieval workflows are usable today; APIs may evolve as the toolkit matures.

## Features

**Ingestion**

- **Document Loading**: PDF, DOCX, XLSX, PPTX and more via MarkItDown
- **LLM Metadata Extraction**: extracts company, doc type, fiscal period using your LLM; fully customisable via YAML
- **Smart Text Splitting**: markdown-aware, recursive, and page-wise chunking strategies; page-wise keeps one chunk per PDF page or PPTX slide and stamps `page_number` on every chunk
- **SHA256 Deduplication**: at both file and chunk level
- **Directory Ingestion**: ingest an entire folder with one call, with optional recursive scan
- **Source Sync**: `rag.sync()` reconciles against local folders and S3, including deletions that plain ingestion never sees

**Retrieval and answers**

- **Multiple Embedding Providers**: Ollama, OpenAI, OpenRouter, HuggingFace, Google, FastEmbed
- **Qdrant Vector Store**: dense, sparse, and hybrid search
- **Advanced Retrieval**: similarity, MMR, and hybrid search with metadata filtering
- **Reranking**: optional cross-encoder second stage, local and API-key-free by default
- **Grounded Answers**: `rag.query()` returns cited answers and refuses when the documents come up short, with `aquery()` for async callers

**Operating it**

- **Evaluation**: recall@k, MRR and config sweeps against your own golden set, so tuning is measured rather than guessed
- **MCP Server**: `ragwire mcp serve` exposes a collection to Claude Desktop, Claude Code and Cursor
- **CLI**: `ragwire ingest`, `sync`, `eval` and `mcp serve`
- **Env Var Substitution**: use `${VAR}` in `config.yaml` for secrets

## Architecture

<p align="center">
  <img src="https://raw.githubusercontent.com/laxmimerit/RAGWire/main/assets/ragwire-system-architecture.png" alt="RAGWire system architecture showing document ingestion and query retrieval workflows" width="100%"/>
</p>

RAGWire is coordinated by the `RAGWire` core and configured through YAML. Documents move through conversion, deduplication, chunking, metadata extraction, embeddings, and Qdrant storage. Queries can use explicit or LLM-extracted metadata filters before dense, sparse, or hybrid retrieval returns the most relevant chunks.

### Document Ingestion Flow

<p align="center">
  <img src="https://raw.githubusercontent.com/laxmimerit/RAGWire/main/assets/ragwire-ingestion-flow.png" alt="RAGWire document ingestion flow from source files to searchable chunks in Qdrant" width="100%"/>
</p>

The ingestion path is safe to re-run: SHA256 file hashes skip previously stored documents. New files are converted to Markdown, split into overlapping chunks, enriched with structured LLM metadata, embedded, and stored in Qdrant with dense and optional sparse vectors.

## Installation

```bash
pip install ragwire

# With Ollama support (local, no API key)
pip install "ragwire[ollama]"

# With OpenRouter support (LLM + embeddings; requires Python >= 3.10)
pip install "ragwire[openrouter]"

# Optional capabilities
pip install "ragwire[rerank]"   # local cross-encoder reranking, no API key
pip install "ragwire[cohere]"   # hosted reranking
pip install "ragwire[mcp]"      # MCP server for Claude Desktop / Code / Cursor
pip install "ragwire[s3]"       # S3 source connector for rag.sync()

# Everything
pip install "ragwire[all]"
```

Nothing above is needed for the core install. Reranking, MCP and S3 are opt-in, and evaluation needs no extra package at all.

## Quick Start

```python
from ragwire import RAGWire

rag = RAGWire("config.yaml")

# Ingest files. SHA256 deduplication makes this safe to re-run.
stats = rag.ingest_documents(["data/Apple_10k_2025.pdf", "data/Microsoft_10k_2025.pdf"])
print(f"Processed: {stats['processed']}, Skipped: {stats['skipped']}, Chunks: {stats['chunks_created']}")

# Or ingest an entire directory
stats = rag.ingest_directory("data/", recursive=True)

# Basic retrieval returns a list of LangChain Document objects
results = rag.retrieve("What is the total revenue?", top_k=5)
for doc in results:
    print(doc.page_content[:300])
    print(doc.metadata["company_name"])   # str, lowercased, e.g. "apple"
    print(doc.metadata["fiscal_year"])    # int, e.g. 2025
    print(doc.metadata["file_name"])      # str, e.g. "Apple_10k_2025.pdf"

# Retrieval with explicit metadata filters
results = rag.retrieve(
    "What is the net income?",
    filters={"company_name": "apple", "fiscal_year": 2025}  # pass year as int
)

# A list gives OR logic within a field, matching any of the listed values
results = rag.retrieve("Compare revenue trends", filters={"fiscal_year": [2023, 2024, 2025]})

# Agent-controlled filtering (recommended for AI agents)
filters = rag.extract_filters("Apple's revenue in 2025")
# → {"company_name": "apple", "fiscal_year": 2025} or None
results = rag.retrieve("Apple's revenue in 2025", filters=filters)
```

### Grounded Answers

`retrieve()` returns chunks. `query()` returns an answer with the sources it came from, and refuses rather than falling back on the model's own knowledge.

```python
answer = rag.query("What was Apple's net income in fiscal 2025?")

print(answer.text)          # "Apple reported net income of $93.7 billion [1]."
print(answer.sources)       # ["Apple_10k_2025.pdf"]
print(answer.confidence)    # fraction of sentences carrying a citation

# An Answer is falsy when the documents did not support one
if not answer:
    print("Not in the collection")

# Async, for anything serving concurrent users
answer = await rag.aquery("What was net income?")
```

### Keeping the Collection Current

Ingestion only ever adds, so a file deleted at the source keeps answering queries forever. `sync()` reconciles instead.

```yaml
sources:
  - type: local
    path: "./documents"
    recursive: true
```

```python
stats = rag.sync(dry_run=True)   # see what would change
stats = rag.sync()               # ingest new, replace changed, delete gone
```

### Command Line

```bash
ragwire ingest ./documents --recursive
ragwire sync --dry-run
ragwire eval golden.yaml --compare-rerank
ragwire mcp serve --config config.yaml    # expose the collection to Claude Desktop
```

### Query & Retrieval Flow

<p align="center">
  <img src="https://raw.githubusercontent.com/laxmimerit/RAGWire/main/assets/ragwire-retrieval-flow.png" alt="RAGWire query and retrieval flow with metadata filters and hybrid search" width="100%"/>
</p>

At query time, callers can pass filters directly or enable LLM-assisted filter extraction grounded in values already stored in Qdrant. The query is then resolved through similarity, MMR, or dense-plus-sparse hybrid search and returned as top-ranked LangChain `Document` chunks for a RAG application or agent.

## Configuration

Create a minimal `config.yaml`:

```yaml
embeddings:
  provider: "ollama"
  model: "qwen3-embedding:0.6b"
  base_url: "http://localhost:11434"

llm:
  provider: "ollama"
  model: "qwen3.5:9b"
  num_ctx: 16384

vectorstore:
  url: "http://localhost:6333"
  collection_name: "rag_documents"
  use_sparse: true

retriever:
  search_type: "hybrid"
  top_k: 5
```

Copy `config.example.yaml` to `config.yaml` for the full template. Secrets can be injected via environment variables:

```yaml
vectorstore:
  url: "https://your-cluster.qdrant.io"
  api_key: "${QDRANT_API_KEY}"

llm:
  provider: "openai"
  model: "gpt-5.4-nano"
  api_key: "${OPENAI_API_KEY}"
```

Full example:

```yaml
embeddings:
  provider: "ollama"
  model: "qwen3-embedding:0.6b"
  base_url: "http://localhost:11434"

llm:
  provider: "ollama"
  model: "qwen3.5:9b"
  num_ctx: 16384

vectorstore:
  url: "http://localhost:6333"
  collection_name: "my_docs"
  use_sparse: true

retriever:
  search_type: "hybrid"
  top_k: 5
  auto_filter: false   # set true to enable LLM-based filter extraction from every query
```

### Optional blocks

Each of these is entirely optional. Omit it and the feature is off.

```yaml
retriever:
  top_k: 5
  rerank:                              # needs: pip install "ragwire[rerank]"
    provider: "cross_encoder"          # or "cohere"
    model: "BAAI/bge-reranker-base"
    fetch_k: 25                        # candidates scored, default max(4 * top_k, 20)

generation:                            # controls rag.query() only
  max_context_chars: 12000             # budget for the source block sent to the LLM

sources:                               # used by rag.sync()
  - type: local
    path: "./documents"
    recursive: true
  - type: s3                           # needs: pip install "ragwire[s3]"
    bucket: "my-filings"
    prefix: "2026/"
```

See the [API reference](https://laxmimerit.github.io/RAGWire/api_reference/) for every key.

## Embedding Providers

```yaml
# Ollama (local)
embeddings:
  provider: "ollama"
  model: "qwen3-embedding:0.6b"

# OpenAI
embeddings:
  provider: "openai"
  model: "text-embedding-3-small"

# OpenRouter (free-tier models available)
embeddings:
  provider: "openrouter"
  model: "nvidia/llama-nemotron-embed-vl-1b-v2:free"
  api_key: "${OPENROUTER_API_KEY}"

# HuggingFace (local)
embeddings:
  provider: "huggingface"
  model_name: "sentence-transformers/all-MiniLM-L6-v2"

# Google
embeddings:
  provider: "google"
  model: "models/embedding-001"
```

## Examples

Start with the basic examples, then move into app and agent tutorials:

| Example | Path |
|---|---|
| Basic ingestion and retrieval | [`examples/basic_usage.py`](examples/basic_usage.py) |
| Custom metadata extraction | [`examples/custom_metadata_usage.py`](examples/custom_metadata_usage.py) |
| RAG agent helper | [`examples/rag_agent.py`](examples/rag_agent.py) |
| Golden set for evaluation | [`examples/golden.example.yaml`](examples/golden.example.yaml) |
| Tutorial series overview | [`examples/tutorials/00_series_overview.md`](examples/tutorials/00_series_overview.md) |
| FastAPI production app | [`examples/tutorials/15_fastapi_production_app.md`](examples/tutorials/15_fastapi_production_app.md) |
| LangGraph RAG pipeline | [`examples/tutorials/05_langgraph_rag_pipeline.md`](examples/tutorials/05_langgraph_rag_pipeline.md) |
| Chainlit RAG chatbot | [`examples/tutorials/03_chainlit_rag_chatbot.md`](examples/tutorials/03_chainlit_rag_chatbot.md) |

## Production Notes

- Use Qdrant Cloud or a persistent local Qdrant volume for real projects.
- Keep `force_recreate: false` after initial setup to avoid accidental collection resets.
- Store API keys in environment variables and reference them as `${VAR}` in YAML.
- Tune `chunk_size`, `chunk_overlap`, `top_k`, and `search_type` for your document type.
- Use metadata filters for high-precision retrieval over multi-company, multi-year, or multi-domain collections.
- Enable `use_sparse: true` with `fastembed` when keyword matching matters alongside semantic search.
- **Measure before you tune.** Write twenty real questions into a golden set and run `ragwire eval golden.yaml`. Chunk size and `top_k` advice, including the advice above, is a guess until it is checked against your own corpus.
- **Turn on reranking, then prove it helped.** `ragwire eval golden.yaml --compare-rerank` runs both ways. Unchanged recall with improved MRR is the expected shape: reranking reorders the candidate pool rather than widening it.
- **Prefer `sync()` over repeated `ingest_directory()`** for anything scheduled. Ingestion only ever adds, so a document deleted at the source keeps answering queries forever. Run `ragwire sync --dry-run` the first time.
- **Log `answer.confidence` and `answer.refused`** in user-facing apps. A run of refusals usually means a retrieval or filtering problem, not a generation one.

## Security & Privacy

RAGWire can run with local Ollama models for teams that do not want document text sent to hosted LLM providers. If you configure hosted LLM or embedding providers, the relevant document text or query text is sent to those providers for metadata extraction, embeddings, answer generation, or filter extraction. Keep secrets out of source control by using environment variables in `config.yaml`.

A fully local setup is possible end to end: Ollama for the LLM and embeddings, `cross_encoder` for reranking, and a local or self-hosted Qdrant. Reranking in particular defaults to a local model precisely so that turning it on does not push document text to a third party.

`rag.query()` sends retrieved chunk text to whichever LLM you configured. The MCP server exposes your collection to whatever client you connect it to, so treat that connection with the same care as any other data access.

## How RAGWire Fits

RAGWire is a composable Python toolkit, not a hosted RAG platform and not a full application framework. Use it when you want direct control over document ingestion, metadata design, Qdrant storage, and retrieval behavior while still avoiding boilerplate.

## Component Usage

```python
from ragwire import (
    MarkItDownLoader,
    get_splitter,
    get_markdown_splitter,
    get_embedding,
    QdrantStore,
    MetadataExtractor,
    hybrid_search,
    mmr_search,
)

# Load a document
loader = MarkItDownLoader()
result = loader.load("document.pdf")

# Split text
splitter = get_markdown_splitter(chunk_size=10000, chunk_overlap=2000)
chunks = splitter.split_text(result["text_content"])

# Or one chunk per page (set splitter.strategy: "page" to use this in the pipeline).
# Pages come from the format itself: PDF pages, PPTX slides, "<!-- pagebreak -->"
# markers in text files, or markdown/HTML headings.
from ragwire import PageLoader, PageSplitter

loader = PageLoader()
result = loader.load("report.pdf")
pages = PageSplitter().split(
    result["text_content"], pages=result["pages"], file_type=result["file_type"]
)
# each page: {"text", "page_number", "page_label", "page_total"}

# Embeddings
embedding = get_embedding({"provider": "ollama", "model": "qwen3-embedding:0.6b"})

# Vector store
store = QdrantStore(config={"url": "http://localhost:6333"}, embedding=embedding)
store.set_collection("my_collection")
vectorstore = store.get_store()
```

The newer subsystems are composable in the same way. Each takes what it needs and holds no hidden dependency on `RAGWire`:

```python
from ragwire.eval import GoldenSet, evaluate, sweep
from ragwire.generation import AnswerGenerator
from ragwire.mcp import search_documents, get_filter_context
from ragwire.retriever import get_reranker
from ragwire.sources import LocalSource, REGISTRY, Source

# Answer from documents you selected yourself
generator = AnswerGenerator(rag.llm, max_context_chars=20000)
answer = generator.generate("What was net income?", my_documents)

# Rerank a candidate list from anywhere
reranker = get_reranker({"provider": "cross_encoder"})
best = reranker.rerank("net income", candidates, top_n=5)

# Agent-ready text without running an MCP server
context = get_filter_context(rag, "apple revenue")
```

## Package Structure

```
ragwire/
├── core/          # Config loader + RAGWire orchestrator
├── loaders/       # MarkItDown converter + page-aware loader (pypdf, python-pptx)
├── processing/    # Text splitters + SHA256 hashing
├── metadata/      # Pydantic schema + LLM extractor
├── embeddings/    # Multi-provider embedding factory
├── vectorstores/  # Qdrant wrapper with hybrid search
├── retriever/     # Similarity, MMR, hybrid retrieval and reranking
├── generation/    # Grounded answers with citations
├── eval/          # Golden sets, recall@k, MRR, config sweeps
├── sources/       # Local and S3 connectors for rag.sync()
├── mcp/           # MCP server and tools
├── cli.py         # The ragwire command
└── utils/         # Logging
```

## Troubleshooting

| Error | Fix |
|-------|-----|
| Qdrant connection refused | `docker run -p 6333:6333 qdrant/qdrant` |
| `markitdown[pdf]` missing | `pip install "markitdown[pdf]"` |
| Ollama model not found | `ollama pull <model-name>` |
| `fastembed` missing | `pip install fastembed` (needed for hybrid search) |
| Embedding dimension mismatch | Set `force_recreate: true` in config once, then back to `false` |

## Contributing

Contributions are welcome. Please open an issue for bugs, feature requests, provider integrations, or documentation improvements before larger changes.

## License

MIT © 2026 [KGP Talkie Private Limited](https://kgptalkie.com)

## Links

<p align="center">
  <a href="https://laxmimerit.github.io/RAGWire/">
    <img src="https://img.shields.io/badge/📖%20Documentation-Visit%20Docs-2ea44f?style=for-the-badge&logo=gitbook&logoColor=white" alt="Documentation"/>
  </a>
  &nbsp;
  <a href="https://github.com/laxmimerit/ragwire">
    <img src="https://img.shields.io/badge/⭐%20GitHub-Star%20the%20Repo-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"/>
  </a>
  &nbsp;
  <a href="https://youtube.com/kgptalkie">
    <img src="https://img.shields.io/badge/▶%20YouTube-KGP%20Talkie-FF0000?style=for-the-badge&logo=youtube&logoColor=white" alt="YouTube"/>
  </a>
</p>

- 🌐 Website: [kgptalkie.com](https://kgptalkie.com)
- 📖 Docs: [laxmimerit.github.io/RAGWire](https://laxmimerit.github.io/RAGWire/)
- 💻 GitHub: [github.com/laxmimerit/ragwire](https://github.com/laxmimerit/ragwire)
- 📧 Email: udemy@kgptalkie.com
