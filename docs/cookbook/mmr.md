# Get Diverse Results with MMR

By default, `retrieve()` returns the top-N most similar chunks — which can be nearly identical if they come from the same page. Use MMR (Maximal Marginal Relevance) to get results spread across different parts of the document.

## Quick Setup via Config

```yaml
retriever:
  search_type: "mmr"
  top_k: 5
```

```python
from ragwire import RAGWire

rag = RAGWire("config.yaml")
results = rag.retrieve("Apple revenue breakdown")
```

## Fine-Grained Control with `mmr_search`

```python
from ragwire import mmr_search

results = mmr_search(
    rag.vectorstore,
    query="Apple revenue breakdown",
    k=5,
    fetch_k=20,      # fetch 20 candidates, pick the 5 most diverse
    lambda_mult=0.3, # lean towards diversity (0.0 = max diverse, 1.0 = max relevant)
)

for doc in results:
    print(doc.metadata.get("chunk_index"), doc.page_content[:100])
```

## Parameter Guide

| Parameter | Default | Effect |
|---|---|---|
| `k` | `5` | Number of results to return |
| `fetch_k` | `20` | Candidate pool size. Higher = better diversity coverage |
| `lambda_mult` | `0.5` | `0.0` = maximize diversity, `1.0` = maximize relevance |

**Recommended starting values:**

- `fetch_k = 3–4x k` (e.g. `k=5, fetch_k=20`)
- `lambda_mult = 0.3–0.5` for a balanced diversity/relevance trade-off

## When to Use MMR

Use MMR when:

- Your documents have long repetitive sections (boilerplate in contracts, repeated tables in financial reports)
- Similarity search keeps returning chunks from the same page
- You want results that cover different aspects of a topic

Stick with `similarity` when you need the absolute highest-relevance matches and repetition is not a concern.
