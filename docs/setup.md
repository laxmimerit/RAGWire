# RAGWire — Setup Guide

This guide covers the generic setup of RAGWire, prerequisites, and the project structure. Provider-specific configuration is covered in the individual provider docs.

## Prerequisites

- Python 3.9 or higher
- [Docker](https://www.docker.com/) (for running Qdrant locally)

## 1. Install RAGWire

Install the base package:

```bash
pip install ragwire
```

Install with a specific provider:

```bash
pip install "ragwire[ollama]"       # Ollama (local, no API key)
pip install "ragwire[openai]"       # OpenAI
pip install "ragwire[google]"       # Google Gemini
pip install "ragwire[huggingface]"  # HuggingFace (local)
pip install "ragwire[groq]"         # Groq
pip install "ragwire[anthropic]"    # Anthropic Claude
pip install "ragwire[all]"          # Everything
```

For hybrid search (dense + sparse), also install:

```bash
pip install fastembed
```

## 2. Start Qdrant

RAGWire uses [Qdrant](https://qdrant.tech/) as its vector database. Run it locally with Docker:

```bash
docker run -d -p 6333:6333 --name qdrant qdrant/qdrant
```

Verify it is running:

```bash
curl http://localhost:6333/healthz
# {"title":"qdrant - vector search engine"}
```

## 3. Configure

Copy the example config and edit it for your provider:

```bash
cp config.example.yaml config.yaml
```

The minimal required sections are:

```yaml
embeddings:
  provider: "..."
  model: "..."

llm:
  provider: "..."
  model: "..."

vectorstore:
  url: "http://localhost:6333"
  collection_name: "my_docs"
  use_sparse: true

retriever:
  search_type: "hybrid"
  top_k: 5
```

See the provider docs for the exact values to fill in.

### Using environment variables in config

Any string value in `config.yaml` can reference an environment variable using `${VAR}` syntax:

```yaml
vectorstore:
  url: "https://your-cluster.qdrant.io"
  api_key: "${QDRANT_API_KEY}"

llm:
  provider: "openai"
  model: "gpt-4o-mini"
  api_key: "${OPENAI_API_KEY}"
```

RAGWire resolves these at startup (after loading `.env` via `python-dotenv`). If a variable is not set, the placeholder is kept and a warning is logged.

## 4. Place Documents

Put your PDF (or DOCX, XLSX, PPTX, TXT, MD) files in `examples/data/`:

```
examples/
└── data/
    └── Apple_10k_2025.pdf
```

## 5. Run

```bash
python examples/basic_usage.py
```

## Project Structure

```
ragwire/
├── core/           Config loader + RAGWire orchestrator
├── loaders/        MarkItDown document converter
├── processing/     Text splitters + SHA256 hashing
├── metadata/       Pydantic schema + LLM extractor
├── embeddings/     Multi-provider embedding factory
├── vectorstores/   Qdrant wrapper with hybrid search
├── retriever/      Similarity, MMR, hybrid retrieval
└── utils/          Logging

config.yaml         Your active configuration (not committed)
config.example.yaml Template to copy from
examples/
└── basic_usage.py  End-to-end pipeline example
```

## Splitter Strategies

| Strategy | Best For |
|---|---|
| `markdown` | PDFs, DOCX — MarkItDown converts to markdown, splits on headers |
| `recursive` | Plain text, generic documents |

## Search Types

| Search Type | Description | Requires |
|---|---|---|
| `similarity` | Dense vector cosine search | — |
| `mmr` | Maximal Marginal Relevance — diverse results | — |
| `hybrid` | Dense + sparse (keyword) combined | `use_sparse: true` + `fastembed` |

## Troubleshooting

| Error | Fix |
|---|---|
| Qdrant connection refused | `docker run -p 6333:6333 qdrant/qdrant` |
| `markitdown[pdf]` missing | `pip install "markitdown[pdf]"` |
| Embedding dimension mismatch | Set `force_recreate: true` in config once, then back to `false` |
| Collection has no sparse vectors | Set `force_recreate: true` in config once, then back to `false` |
| `fastembed` missing | `pip install fastembed` |
