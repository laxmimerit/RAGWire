# ragwire — Setup Complete

## Package Structure

```
rag-setup/
├── ragwire/                        # Main package
│   ├── core/
│   │   ├── config.py               # YAML + ENV configuration loader
│   │   ├── rag.py             # Main RAG pipeline orchestrator
│   │   └── __init__.py
│   ├── loaders/
│   │   ├── markitdown_loader.py    # Document conversion (PDF, DOCX, XLSX, etc.)
│   │   └── __init__.py
│   ├── processing/
│   │   ├── splitter.py             # Markdown, recursive, and code text splitters
│   │   ├── hashing.py              # SHA256 hashing for deduplication
│   │   └── __init__.py
│   ├── metadata/
│   │   ├── schema.py               # Pydantic metadata schema
│   │   ├── extractor.py            # LLM-based metadata extraction
│   │   └── __init__.py
│   ├── embeddings/
│   │   ├── factory.py              # Multi-provider embedding factory
│   │   └── __init__.py
│   ├── vectorstores/
│   │   ├── qdrant_store.py         # Qdrant vector store wrapper
│   │   └── __init__.py
│   ├── retriever/
│   │   ├── hybrid.py               # Similarity, MMR, and hybrid retrieval
│   │   └── __init__.py
│   ├── utils/
│   │   ├── logging.py              # Logging utilities (colored + file)
│   │   └── __init__.py
│   └── __init__.py                 # Public API exports
├── assets/
│   └── ragwire.png                 # Package logo
├── examples/
│   ├── basic_usage.py              # End-to-end example
│   └── data/                       # Place PDF/DOCX files here
├── pyproject.toml
├── requirements.txt
├── config.yaml                     # Your active configuration
├── config.example.yaml             # Annotated configuration template
├── .env                            # Environment variables
├── .env.example                    # Environment variables template
└── README.md
```

## What's Implemented

### Core Components

| Component | File | Description |
|-----------|------|-------------|
| Config | `core/config.py` | YAML loading, dot-notation access, ENV overrides |
| Pipeline | `core/rag.py` | Full ingest → retrieve orchestration |
| Loader | `loaders/markitdown_loader.py` | PDF, DOCX, XLSX, PPTX → markdown |
| Splitter | `processing/splitter.py` | Markdown, recursive, code splitters |
| Hashing | `processing/hashing.py` | SHA256 for file and chunk deduplication |
| Metadata | `metadata/extractor.py` | LLM-based extraction (once per document, from first chunk) |
| Embeddings | `embeddings/factory.py` | Ollama, OpenAI, HuggingFace, Google, FastEmbed |
| Vector Store | `vectorstores/qdrant_store.py` | Qdrant wrapper, hybrid search, collection management |
| Retriever | `retriever/hybrid.py` | Similarity, MMR, hybrid (dense + sparse) |
| Logging | `utils/logging.py` | Colored console + file logging |

## Quick Start

### 1. Install

```bash
cd D:\Packages\rag-setup

# Ollama (local, no API key needed)
pip install -e ".[ollama]"

# All providers
pip install -e ".[all]"
```

### 2. Start Services

```bash
# Qdrant (Docker)
docker run -p 6333:6333 qdrant/qdrant

# Ollama — pull models
ollama pull qwen3-embedding:0.6b
ollama pull qwen3.5:9b
```

### 3. Configure

`config.yaml` is already set up for local Ollama. Key settings:

```yaml
embeddings:
  provider: "ollama"
  model: "qwen3-embedding:0.6b"

llm:
  provider: "ollama"
  model: "qwen3.5:9b"
  temperature: 0.0
  num_ctx: 16384

vectorstore:
  url: "http://localhost:6333"
  collection_name: "financial_docs"
```

### 4. Run

```bash
# Place PDFs in examples/data/ then:
python examples/basic_usage.py
```

## Usage

### Pipeline (high-level)

```python
from ragwire import RAGWire

rag = RAGWire("config.yaml")

stats = rag.ingest_documents(["data/Apple_10k_2025.pdf"])
print(f"Chunks created: {stats['chunks_created']}")

results = rag.retrieve("What is Apple's total revenue?", top_k=5)
for doc in results:
    print(doc.metadata.get("company_name"), doc.page_content[:200])
```

### Components (low-level)

```python
from ragwire import MarkItDownLoader, get_splitter, get_embedding, QdrantStore

loader = MarkItDownLoader()
result = loader.load("document.pdf")

splitter = get_splitter(chunk_size=10000)
chunks = splitter.split_text(result["text_content"])

embedding = get_embedding({"provider": "ollama", "model": "qwen3-embedding:0.6b"})

store = QdrantStore(config={"url": "http://localhost:6333"}, embedding=embedding)
store.set_collection("my_docs")
vectorstore = store.get_store()
```

## Embedding Providers

```yaml
# Ollama (local)
embeddings:
  provider: "ollama"
  model: "qwen3-embedding:0.6b"
  base_url: "http://localhost:11434"

# OpenAI
embeddings:
  provider: "openai"
  model: "text-embedding-3-small"

# HuggingFace (local)
embeddings:
  provider: "huggingface"
  model_name: "sentence-transformers/all-MiniLM-L6-v2"

# Google
embeddings:
  provider: "google"
  model: "models/embedding-001"
```

## Search Strategies

```yaml
# Similarity (default)
retriever:
  search_type: "similarity"
  top_k: 5

# Hybrid — requires use_sparse: true + pip install fastembed
retriever:
  search_type: "hybrid"
  top_k: 5
vectorstore:
  use_sparse: true

# MMR — diverse results
retriever:
  search_type: "mmr"
  top_k: 5
```

## Troubleshooting

| Error | Fix |
|-------|-----|
| `qdrant connection refused` | `docker run -p 6333:6333 qdrant/qdrant` |
| `markitdown[pdf]` missing | `pip install "markitdown[pdf]"` |
| Ollama model not found | `ollama pull <model-name>` |
| `fastembed` missing (hybrid) | `pip install fastembed` |
| Stale collection dimensions | Set `force_recreate: true` in config.yaml, run once, then set back to `false` |
