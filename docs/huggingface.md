# RAGWire with HuggingFace

Use HuggingFace sentence-transformers for local embeddings — no API key, runs on CPU or GPU.

> HuggingFace does not provide a hosted chat LLM in RAGWire. Pair it with Ollama, OpenAI, Groq, or Anthropic for the metadata extraction LLM.

## Prerequisites

- RAGWire installed: `pip install "ragwire[huggingface]"`
- Qdrant running: `docker run -d -p 6333:6333 qdrant/qdrant`

## 1. Install Dependencies

```bash
# HuggingFace for embeddings + Ollama for LLM (fully local, no cost)
pip install "ragwire[huggingface]" "ragwire[ollama]"

# Or with OpenAI for LLM
pip install "ragwire[huggingface]" "ragwire[openai]"

pip install fastembed               # For hybrid search
```

## 2. Configuration

### HuggingFace Embeddings + Ollama LLM (fully local)

```yaml
embeddings:
  provider: "huggingface"
  model_name: "sentence-transformers/all-MiniLM-L6-v2"   # 384-dim, fast
  # model_name: "BAAI/bge-large-en-v1.5"                 # 1024-dim, higher quality
  model_kwargs:
    device: "cpu"     # "cuda" for GPU, "mps" for Apple Silicon
  encode_kwargs:
    normalize_embeddings: true

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

### HuggingFace Embeddings + OpenAI LLM

```yaml
embeddings:
  provider: "huggingface"
  model_name: "sentence-transformers/all-MiniLM-L6-v2"
  model_kwargs:
    device: "cpu"

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
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | Fast, lightweight, good general purpose |
| `sentence-transformers/all-mpnet-base-v2` | 768 | Better quality, still fast |
| `BAAI/bge-large-en-v1.5` | 1024 | High quality, larger model |
| `BAAI/bge-m3` | 1024 | Multilingual, very strong |
| `intfloat/e5-large-v2` | 1024 | Strong retrieval performance |

## GPU Acceleration

```yaml
embeddings:
  provider: "huggingface"
  model_name: "BAAI/bge-large-en-v1.5"
  model_kwargs:
    device: "cuda"     # NVIDIA GPU
    # device: "mps"    # Apple Silicon
```

## Notes

- Models are downloaded from HuggingFace Hub on first use and cached locally (`~/.cache/huggingface/`).
- If you change the model after ingestion, set `force_recreate: true` once to rebuild the collection (dimensions may differ).
