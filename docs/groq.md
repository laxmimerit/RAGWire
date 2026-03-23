# RAGWire with Groq

Groq provides ultra-fast LLM inference. Use it as the metadata extraction LLM paired with any embedding provider.

> Groq does not provide embedding models. You need a separate embedding provider — Ollama (local/free) or OpenAI are recommended.

## Prerequisites

- Groq API key — [console.groq.com](https://console.groq.com)
- RAGWire installed: `pip install "ragwire[groq]"`
- Qdrant running: `docker run -d -p 6333:6333 qdrant/qdrant`

## 1. Install Dependencies

```bash
# Groq for LLM + Ollama for embeddings (fully local embeddings, no cost)
pip install "ragwire[groq]" "ragwire[ollama]"

# Or with OpenAI for embeddings
pip install "ragwire[groq]" "ragwire[openai]"

pip install fastembed               # For hybrid search
```

## 2. Set API Key

```bash
# Linux / macOS
export GROQ_API_KEY="gsk_..."

# Windows (PowerShell)
$env:GROQ_API_KEY="gsk_..."
```

Or add it to a `.env` file at the project root:

```
GROQ_API_KEY=gsk_...
```

## 3. Configuration

### Groq LLM + Ollama Embeddings (recommended — free embeddings)

```yaml
embeddings:
  provider: "ollama"
  model: "nomic-embed-text"
  base_url: "http://localhost:11434"

llm:
  provider: "groq"
  model: "qwen/qwen3-32b"            # Latest — strong quality, thinking mode support
  # model: "llama-3.3-70b-versatile" # High quality alternative

vectorstore:
  url: "http://localhost:6333"
  collection_name: "my_docs"
  use_sparse: true
  force_recreate: false

retriever:
  search_type: "hybrid"
  top_k: 5
```

### Groq LLM + OpenAI Embeddings

```yaml
embeddings:
  provider: "openai"
  model: "text-embedding-3-small"

llm:
  provider: "groq"
  model: "qwen/qwen3-32b"            # 131K context, thinking mode support

vectorstore:
  url: "http://localhost:6333"
  collection_name: "my_docs"
  use_sparse: true
  force_recreate: false

retriever:
  search_type: "hybrid"
  top_k: 5
```

## 4. Python Usage

```python
from ragwire import RAGWire

rag = RAGWire("config.yaml")

# Ingest
stats = rag.ingest_documents(["data/Apple_10k_2025.pdf"])
print(f"Chunks created: {stats['chunks_created']}")

# Retrieve
results = rag.retrieve("What is Apple's total revenue?", top_k=5)
for doc in results:
    print(doc.metadata.get("company_name"), doc.page_content[:200])
```

## 5. Run the Example

```bash
python examples/basic_usage.py
```

## Available Models

| Model | Context | Speed | Notes |
|---|---|---|---|
| `qwen/qwen3-32b` | 131K | — | Latest — recommended, thinking mode support |
| `llama-3.3-70b-versatile` | 131K | ~280 tok/s | High quality alternative |
| `llama-3.1-8b-instant` | 131K | ~560 tok/s | Ultra-fast |
| `openai/gpt-oss-120b` | 131K | ~500 tok/s | High quality |
| `openai/gpt-oss-20b` | 131K | ~1000 tok/s | Fastest large model |

Full list: [console.groq.com/docs/models](https://console.groq.com/docs/models)

## Notes

- The API key can also be passed directly in config: `api_key: "gsk_..."` — but environment variables are preferred.
- Groq's free tier has generous rate limits — well suited for development and testing.
