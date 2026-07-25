# Reranking

Reranking is usually the single largest jump in answer quality you can get from a RAG system, and it takes four lines of config.

## Why it helps

Dense retrieval embeds your query and your documents separately, then compares the two vectors. Nothing ever reads the query and the document together. That makes it fast enough to search millions of chunks, but it also means the top result is the chunk whose *summary* is closest to your query, not the chunk that actually answers it.

A reranker is a different kind of model. It takes the query and one candidate chunk as a single input and scores that pair directly. It is far more accurate and far too slow to run over a whole collection, so the standard arrangement is:

1. Retrieve a wide pool of candidates cheaply (25 chunks instead of 5)
2. Score every candidate with the reranker
3. Keep the best 5

You pay for one extra model pass over 25 short texts, and in exchange the chunks you send to your LLM are the ones that genuinely answer the question.

!!! note "Reranking is not MMR"
    [MMR](mmr.md) reorders results for *diversity*, and will deliberately push a relevant chunk down the list to avoid returning two similar ones. Reranking reorders for *relevance*. They solve different problems and can be used together.

## Local reranking, no API key

The default provider runs a small cross-encoder on your machine. It works on CPU and needs no account anywhere.

```bash
pip install ragwire[rerank]
```

```yaml title="config.yaml"
retriever:
  search_type: "hybrid"
  top_k: 5
  rerank:
    provider: "cross_encoder"
    model: "BAAI/bge-reranker-base"
    fetch_k: 25
```

That is the whole change. Your Python code does not move:

```python
from ragwire import RAGWire

rag = RAGWire("config.yaml")
results = rag.retrieve("What drove the change in operating margin?")
```

The model downloads on first use, roughly 1 GB for `bge-reranker-base`, and is cached afterwards. Nothing is downloaded when you only ingest, so ingestion scripts are unaffected.

## Hosted reranking with Cohere

If you would rather not run a model locally, Cohere's rerank endpoint is a drop-in swap:

```bash
pip install ragwire[cohere]
```

```yaml title="config.yaml"
retriever:
  top_k: 5
  rerank:
    provider: "cohere"
    model: "rerank-v3.5"
    fetch_k: 25
```

```bash title=".env"
COHERE_API_KEY=your-key-here
```

## Settings

| Key | Default | What it does |
|---|---|---|
| `provider` | `cross_encoder` | `cross_encoder` for local, `cohere` for hosted |
| `model` | `BAAI/bge-reranker-base` | Cohere default is `rerank-v3.5` |
| `fetch_k` | `max(4 * top_k, 20)` | Candidates to fetch and score |
| `enabled` | `true` | Set `false` to switch off without deleting your settings |

Anything else in the block is passed to the provider. `cross_encoder` also accepts `batch_size` and `device`, so `device: "cuda"` moves the model onto a GPU.

### Choosing fetch_k

`fetch_k` is the only setting worth tuning. It controls how many candidates the reranker gets to choose from:

- **Too low** and reranking cannot help. If the right chunk was ranked 30th by the vector store and you only fetch 10, no amount of reranking will surface it.
- **Too high** and every query pays to score chunks that were never plausible.

Start at `25` for `top_k: 5`. If your evaluation shows recall improving as you raise it, your first-stage retrieval is weaker than you thought and the extra candidates are doing real work.

If `fetch_k` is ever set below `top_k`, RAGWire raises it to `top_k`. Otherwise reranking would silently return fewer documents than you asked for.

## Reading the scores

Every reranked document carries its score:

```python
results = rag.retrieve("operating margin drivers")

for doc in results:
    print(f"{doc.metadata['rerank_score']:.3f}  {doc.metadata['source']}")
```

```
 8.214  q3_earnings.pdf
 5.902  q3_earnings.pdf
-1.447  annual_report.pdf
```

Cross-encoder scores are unbounded logits, not probabilities, so compare them to each other rather than to a fixed threshold. A large gap between the first and second result means the top chunk is clearly the best match. A flat spread near zero usually means nothing in your collection answers the question, which is worth knowing before you send it to an LLM.

Cohere scores are normalized to `0.0` to `1.0` instead.

## Measuring whether it helped

Do not take reranking on faith. `retrieve()` takes a per-call override so you can compare against the unreranked baseline with the same config:

```python
reranked = rag.retrieve("operating margin drivers")
baseline = rag.retrieve("operating margin drivers", rerank=False)

print([d.metadata["source"] for d in reranked])
print([d.metadata["source"] for d in baseline])
```

Passing `rerank=True` requires a configured reranker and raises `ValueError` if there is none, which is useful in scripts where silently skipping reranking would invalidate your results.

## What reranking applies to

Reranking is part of `retrieve()`, which is the high-level entry point that honours your whole configured retrieval pipeline.

The lower-level functions `hybrid_search()` and `mmr_search()` are primitives. They do exactly one thing each and deliberately ignore the rerank config, so they stay predictable when you are composing your own pipeline.

## Cost and latency

| Provider | Typical latency for 25 candidates | Cost |
|---|---|---|
| `cross_encoder` on CPU | 200 to 600 ms | Free |
| `cross_encoder` on GPU | 20 to 50 ms | Free |
| `cohere` | 100 to 300 ms | Per API call |

For an interactive chatbot this is usually invisible next to the LLM call that follows it. For a batch job over thousands of queries, a GPU or a smaller `fetch_k` is worth the setup.

## See also

- [Measure Retrieval Quality](evaluation.md) to check whether reranking helped on your corpus
- [Tune Retrieval Quality](tune_retrieval.md) for chunk size, `top_k` and search type
- [Diverse Results (MMR)](mmr.md) for the diversity problem, which reranking does not solve
