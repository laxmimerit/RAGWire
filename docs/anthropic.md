# RAGWire with Anthropic

Use Anthropic Claude as the metadata extraction LLM paired with any embedding provider.

> Anthropic does not provide embedding models. You need a separate embedding provider — Ollama (local/free) or OpenAI are recommended.

## Prerequisites

- Anthropic API key — [console.anthropic.com](https://console.anthropic.com)
- RAGWire installed: `pip install "ragwire[anthropic]"`
- Qdrant running: `docker run -d -p 6333:6333 qdrant/qdrant`

## 1. Install Dependencies

```bash
# Anthropic for LLM + Ollama for embeddings (fully local embeddings, no cost)
pip install "ragwire[anthropic]" "ragwire[ollama]"

# Or with OpenAI for embeddings
pip install "ragwire[anthropic]" "ragwire[openai]"

pip install fastembed               # For hybrid search
```

## 2. Set API Key

```bash
# Linux / macOS
export ANTHROPIC_API_KEY="sk-ant-..."

# Windows (PowerShell)
$env:ANTHROPIC_API_KEY="sk-ant-..."
```

Or add it to a `.env` file at the project root:

```
ANTHROPIC_API_KEY=sk-ant-...
```

## 3. Configuration

### Claude LLM + Ollama Embeddings (recommended — free embeddings)

```yaml
embeddings:
  provider: "ollama"
  model: "nomic-embed-text"
  base_url: "http://localhost:11434"

llm:
  provider: "anthropic"
  model: "claude-haiku-4-5-20251001"   # Fastest, cheapest — ideal for extraction
  # model: "claude-sonnet-4-6"         # Best speed/intelligence balance
  # model: "claude-opus-4-6"           # Most intelligent
  temperature: 0.0

vectorstore:
  url: "http://localhost:6333"
  collection_name: "my_docs"
  use_sparse: true
  force_recreate: false

retriever:
  search_type: "hybrid"
  top_k: 5
```

### Claude LLM + OpenAI Embeddings

```yaml
embeddings:
  provider: "openai"
  model: "text-embedding-3-small"

llm:
  provider: "anthropic"
  model: "claude-haiku-4-5-20251001"   # 200K context, $1/$5 per MTok
  temperature: 0.0

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
from ragwire import RAGPipeline

pipeline = RAGPipeline("config.yaml")

# Ingest
stats = pipeline.ingest_documents(["data/Apple_10k_2025.pdf"])
print(f"Chunks created: {stats['chunks_created']}")

# Retrieve
results = pipeline.retrieve("What is Apple's total revenue?", top_k=5)
for doc in results:
    print(doc.metadata.get("company_name"), doc.page_content[:200])
```

## 5. Run the Example

```bash
python examples/basic_usage.py
```

## Available Models

| Model | Context | Price (in/out per MTok) | Notes |
|---|---|---|---|
| `claude-haiku-4-5-20251001` | 200K | $1 / $5 | Fastest, cheapest — recommended for extraction |
| `claude-sonnet-4-6` | 1M | $3 / $15 | Best speed/intelligence balance |
| `claude-opus-4-6` | 1M | $5 / $25 | Most intelligent |

## Notes

- The API key can also be passed directly in config: `api_key: "sk-ant-..."` — but environment variables are preferred.
- Claude Haiku is the recommended model for metadata extraction — it is fast, cheap, and follows JSON instructions reliably.
