# RAGWire with FastEmbed

FastEmbed is a lightweight, fast embedding library by Qdrant. It runs locally with no API key and is optimized for CPU inference.

> FastEmbed is also used internally for sparse (keyword) vectors when hybrid search is enabled — that part is automatic and requires no extra config.

## Prerequisites

- RAGWire installed: `pip install ragwire`
- FastEmbed installed: `pip install fastembed`
- Qdrant running: `docker run -d -p 6333:6333 qdrant/qdrant`

## 1. Install Dependencies

```bash
# FastEmbed for embeddings + Ollama for LLM (fully local, no cost)
pip install fastembed "ragwire[ollama]"

# Or with OpenAI for LLM
pip install fastembed "ragwire[openai]"
```

## 2. Configuration

### FastEmbed Embeddings + Ollama LLM (fully local)

```yaml
embeddings:
  provider: "fastembed"
  model_name: "BAAI/bge-small-en-v1.5"    # 384-dim, fast and lightweight
  # model_name: "BAAI/bge-base-en-v1.5"   # 768-dim, better quality

llm:
  provider: "ollama"
  model: "qwen3.5:9b"
  base_url: "http://localhost:11434"
  temperature: 0.0
  num_ctx: 16384

vectorstore:
  url: "http://localhost:6333"
  collection_name: "my_docs"
  use_sparse: true
  force_recreate: false

retriever:
  search_type: "hybrid"
  top_k: 5
```

### FastEmbed Embeddings + OpenAI LLM

```yaml
embeddings:
  provider: "fastembed"
  model_name: "BAAI/bge-small-en-v1.5"

llm:
  provider: "openai"
  model: "gpt-4o-mini"
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

## 3. Python Usage

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

## 4. Run the Example

```bash
python examples/basic_usage.py
```

## Recommended Models

| Model | Dimensions | Notes |
|---|---|---|
| `BAAI/bge-small-en-v1.5` | 384 | Default, very fast |
| `BAAI/bge-base-en-v1.5` | 768 | Better quality |
| `BAAI/bge-large-en-v1.5` | 1024 | Best quality |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | Popular alternative |

Full list: [qdrant.github.io/fastembed/examples/Supported_Models](https://qdrant.github.io/fastembed/examples/Supported_Models/)

## Notes

- Models are downloaded and cached locally on first use.
- FastEmbed uses ONNX Runtime — fast on CPU without requiring PyTorch or CUDA.
- If you change the model after ingestion, set `force_recreate: true` once to rebuild the collection.
