# RAGWire with OpenAI

Use OpenAI for both embeddings and the metadata extraction LLM.

## Prerequisites

- OpenAI API key — [platform.openai.com](https://platform.openai.com)
- RAGWire installed: `pip install "ragwire[openai]"`
- Qdrant running: `docker run -d -p 6333:6333 qdrant/qdrant`

## 1. Install Dependencies

```bash
pip install "ragwire[openai]"
pip install fastembed               # For hybrid search
```

## 2. Set API Key

```bash
# Linux / macOS
export OPENAI_API_KEY="sk-..."

# Windows (PowerShell)
$env:OPENAI_API_KEY="sk-..."
```

Or add it to a `.env` file at the project root:

```
OPENAI_API_KEY=sk-...
```

## 3. Configuration

```yaml
embeddings:
  provider: "openai"
  model: "text-embedding-3-small"   # 1536-dim, best price/performance
  # model: "text-embedding-3-large" # 3072-dim, highest quality

llm:
  provider: "openai"
  model: "gpt-5.4-nano"             # Latest — fast, affordable, good for metadata extraction
  # model: "gpt-4o-mini"            # Previous generation
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

## Embedding Model Comparison

| Model | Dimensions | Notes |
|---|---|---|
| `text-embedding-3-small` | 1536 | Best price/performance — recommended |
| `text-embedding-3-large` | 3072 | Highest quality, multilingual |
| `text-embedding-ada-002` | 1536 | Legacy — avoid for new projects |

## Notes

- If you change embedding model after ingestion, set `force_recreate: true` once to rebuild the collection (dimensions will differ).
- The API key can also be passed directly in config: `api_key: "sk-..."` — but environment variables are preferred.
