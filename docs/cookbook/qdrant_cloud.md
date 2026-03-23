# Use Qdrant Cloud (Free Tier)

Run RAGWire against a hosted Qdrant cluster — no Docker, no local storage, fully managed. Qdrant Cloud offers a free tier with 1 GB storage, enough for millions of vectors.

## 1. Create a Free Cluster

1. Sign up at [cloud.qdrant.io](https://cloud.qdrant.io)
2. Create a cluster — select the **Free** tier
3. Copy your **Cluster URL** (e.g. `https://xyz-abc.qdrant.io`) and generate an **API Key**

## 2. Configure RAGWire

Store your credentials in a `.env` file (never commit this):

```bash
# .env
QDRANT_URL=https://xyz-abc.qdrant.io
QDRANT_API_KEY=your-api-key-here
```

Reference them in `config.yaml`:

```yaml
vectorstore:
  url: "${QDRANT_URL}"
  api_key: "${QDRANT_API_KEY}"
  collection_name: "my_docs"
  use_sparse: true
```

RAGWire loads `.env` automatically via `python-dotenv` at startup.

## 3. Run

No other changes needed — the rest of your code is identical to a local setup:

```python
from ragwire import RAGWire

rag = RAGWire("config.yaml")
stats = rag.ingest_directory("data/")
print(f"Processed: {stats['processed']}, Chunks: {stats['chunks_created']}")

results = rag.retrieve("Apple revenue 2025")
for doc in results:
    print(doc.page_content[:200])
```

## Free Tier Limits

| Limit | Value |
|---|---|
| Storage | 1 GB |
| Collections | Unlimited |
| Vectors | ~1M (depends on dimensions) |
| Uptime SLA | None (best effort) |

For production workloads, upgrade to a paid plan or self-host with Docker.

!!! tip "Hybrid search works on Qdrant Cloud"
    Unlike local file storage, Qdrant Cloud fully supports sparse vectors. Set `use_sparse: true` and `search_type: "hybrid"` for the best retrieval quality.
