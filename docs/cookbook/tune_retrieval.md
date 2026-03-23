# Tune Retrieval Quality

Three knobs affect what comes back from retrieval:

| Setting | Default | Effect |
|---|---|---|
| `chunk_size` | `10000` | Larger = more context per chunk; smaller = more precise matches |
| `chunk_overlap` | `2000` | Prevents context being cut at chunk boundaries. ~20% of chunk_size is a good default |
| `top_k` | `5` | More results = broader coverage; fewer = higher precision |

```yaml
splitter:
  chunk_size: 5000      # halve chunk size for more precise retrieval
  chunk_overlap: 1000
  strategy: "markdown"

retriever:
  top_k: 8             # return more candidates
  search_type: "hybrid"
```

**Rule of thumb:**

- Long-form documents (10-Ks, contracts) → larger chunks (`8k–12k`)
- FAQ-style or structured content → smaller chunks (`500–2k`)
- Queries needing breadth (summaries) → higher `top_k`
- Queries needing precision (exact figures) → lower `top_k` + explicit filters

## Search Type Comparison

| Search Type | Best For | Requires |
|---|---|---|
| `similarity` | General-purpose dense retrieval | — |
| `hybrid` | Mixed keyword + semantic queries | `use_sparse: true` + `fastembed` |
| `mmr` | Avoiding repetitive results | — |

## Override at Query Time

`top_k` in config sets the default. Override per query without changing config:

```python
# Broad summary query — return more chunks
results = rag.retrieve("summarize Apple's 2025 annual report", top_k=10)

# Precise fact lookup — return fewer, higher-precision chunks
results = rag.retrieve("Apple net income 2025", top_k=3)
```

!!! tip "Changing chunk_size requires re-ingestion"
    `chunk_size` and `chunk_overlap` only affect new ingestion runs. If you change them, set `force_recreate: true` and re-ingest your documents to apply the new chunking.
