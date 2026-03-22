# RAGWire with Google Gemini

Use Google Gemini for both embeddings and the metadata extraction LLM.

## Prerequisites

- Google AI API key — [aistudio.google.com](https://aistudio.google.com)
- RAGWire installed: `pip install "ragwire[google]"`
- Qdrant running: `docker run -d -p 6333:6333 qdrant/qdrant`

## 1. Install Dependencies

```bash
pip install "ragwire[google]"
pip install fastembed               # For hybrid search
```

## 2. Set API Key

```bash
# Linux / macOS
export GOOGLE_API_KEY="AIza..."

# Windows (PowerShell)
$env:GOOGLE_API_KEY="AIza..."
```

Or add it to a `.env` file at the project root:

```
GOOGLE_API_KEY=AIza...
```

## 3. Configuration

```yaml
embeddings:
  provider: "google"
  model: "models/gemini-embedding-001"   # Stable, recommended for production

llm:
  provider: "google"
  model: "gemini-2.5-flash"              # Best price/performance
  # model: "gemini-2.5-pro"             # Most advanced, deep reasoning
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

## Embedding Model Comparison

| Model | Notes |
|---|---|
| `models/gemini-embedding-001` | Stable, recommended for production |
| `models/gemini-embedding-2-preview` | Newer multimodal embedding (preview) |

## Chat Model Comparison

| Model | Notes |
|---|---|
| `gemini-2.5-flash` | Best price/performance — recommended |
| `gemini-2.5-pro` | Most advanced, deep reasoning |
| `gemini-2.5-flash-lite` | Fastest and most budget-friendly |

## Notes

- Use `provider: "google"` or `provider: "gemini"` — both are accepted.
- The API key can also be passed directly in config: `api_key: "AIza..."` — but environment variables are preferred.
- Free tier has rate limits. For production use, upgrade to a paid plan.
