# RAGWire with Ollama

Ollama lets you run LLMs and embedding models locally — no API key, no cost, no data leaving your machine.

## Prerequisites

- [Ollama](https://ollama.com/) installed and running
- RAGWire installed: `pip install "ragwire[ollama]"`
- Qdrant running: `docker run -d -p 6333:6333 qdrant/qdrant`

## 1. Pull Models

Pull an embedding model and a chat model:

```bash
# Embedding model (small, fast)
ollama pull qwen3-embedding:0.6b

# Chat model for metadata extraction
ollama pull qwen3.5:9b
```

Verify they are available:

```bash
ollama list
```

Other good options:

```bash
# Embeddings
ollama pull nomic-embed-text        # 768-dim, general purpose
ollama pull mxbai-embed-large       # 1024-dim, higher quality

# Chat
ollama pull llama3.2:3b             # Lightweight
ollama pull mistral:7b              # Strong general purpose
```

## 2. Install Dependencies

```bash
pip install "ragwire[ollama]"
pip install fastembed               # For hybrid search
```

## 3. Configuration

```yaml
embeddings:
  provider: "ollama"
  model: "qwen3-embedding:0.6b"
  base_url: "http://localhost:11434"

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

## Notes

- `num_ctx` must be large enough to fit your chunk size. Default `10000` chars → set `num_ctx: 16384` or higher.
- Ollama runs on `http://localhost:11434` by default. If you changed the port, update `base_url`.
- For GPU acceleration, Ollama detects CUDA/Metal automatically — no extra config needed.
