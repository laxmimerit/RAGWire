# Measuring Retrieval Quality

Every tuning knob in RAGWire, chunk size, `top_k`, hybrid search, reranking, is a guess until you measure it. This page shows how to stop guessing.

`ragwire.eval` needs no extra install. It ships with the package.

## The shortest useful version

Write down twenty questions your users actually ask, and which document answers each one:

```yaml title="golden.yaml"
- query: "What was Apple's net income in fiscal 2025?"
  expected: ["apple_10k_2025.pdf"]

- query: "How did Amazon describe AWS growth?"
  expected: ["amazon_10q_q3.pdf", "amazon_10k_2025.pdf"]
```

Then score your pipeline against it:

```python
from ragwire import RAGWire
from ragwire.eval import GoldenSet, evaluate

rag = RAGWire("config.yaml")
golden = GoldenSet.from_file("golden.yaml")

print(evaluate(rag, golden, top_k=5))
```

```
default  (top_k=5, 20 queries)
--------------------------------------------
  recall@5         0.750
  mrr@5            0.612
  hit_rate@5       0.850
  precision@5      0.190

  3 queries retrieved nothing correct:
    - What drove the change in operating margin?
    - Which segment shrank year over year?
    - How is deferred revenue recognised?
```

Those last three lines are the most valuable output on this page. They are the questions your system cannot answer, named specifically, so you can go and look at why.

A full example golden set with every supported option is in [`examples/golden.example.yaml`](https://github.com/laxmimerit/RAGWire/blob/main/examples/golden.example.yaml).

## What the metrics mean

| Metric | Question it answers | Why you care |
|---|---|---|
| **recall@k** | What fraction of the correct documents made the top k? | The ceiling on everything downstream. A chunk that is never retrieved can never be cited. |
| **mrr@k** | How high up was the first correct document? | Sensitive to ordering, which is exactly what reranking changes. |
| **hit_rate@k** | Did anything correct come back at all? | Catches the queries that are simply unanswerable. |
| **precision@k** | What fraction of what you retrieved was useful? | Low precision means you are paying tokens for noise. |

Recall and MRR move independently, and the difference is diagnostic:

- **Recall low, MRR low.** Retrieval is not finding the material. Look at chunking and embeddings, not ranking.
- **Recall high, MRR low.** The right chunks are in the pool but buried. This is precisely what [reranking](reranking.md) fixes.
- **Both high, answers still bad.** Retrieval is fine. The problem is in your prompt or generation step.

## Did reranking actually help?

This is the question the module exists for. `sweep()` runs several retrieval settings against the same golden set:

```python
from ragwire.eval import sweep

print(sweep(rag, golden, {
    "no rerank": {"rerank": False},
    "reranked":  {"rerank": True},
}))
```

```
variant         recall          mrr     hit_rate    precision
-------------------------------------------------------------
no rerank        0.750        0.612        0.850        0.190
reranked    0.750+0.00   0.867+0.26   0.850+0.00   0.190+0.00

Best recall: no rerank (0.750)
```

Read that carefully, because it is the typical result and it is easy to misread. Recall did not move, and it should not have: reranking reorders the candidate pool, it does not widen it. What moved is MRR, by a lot. The correct chunks were always being retrieved; they were just not at the top. After reranking they are, and the LLM sees them first.

Every row after the first shows a delta against the first row, so put your current setup first and your experiment second.

## Other things worth sweeping

Each variant is just keyword arguments for `retrieve()`, so anything that method accepts can be compared:

```python
print(sweep(rag, golden, {
    "top_k=3":  {"top_k": 3},
    "top_k=5":  {"top_k": 5},
    "top_k=10": {"top_k": 10},
}))
```

Raising `top_k` almost always raises recall and lowers precision. The sweep tells you where the trade stops being worth it for your corpus rather than for someone's blog post.

To compare settings that are fixed at startup, like chunk size or embedding model, build one `RAGWire` per config and evaluate each:

```python
from ragwire.eval import EvalResult, SweepResult

results = [
    evaluate(RAGWire("config_small_chunks.yaml"), golden, label="chunk=2000"),
    evaluate(RAGWire("config_large_chunks.yaml"), golden, label="chunk=8000"),
]
print(SweepResult(results))
```

Chunk size changes require re-ingestion into separate collections, so give each config its own `collection_name`.

## Digging into failures

The aggregate number tells you whether to act. The per-query detail tells you what to do:

```python
result = evaluate(rag, golden, top_k=5)

for query_result in result.failures:
    print(query_result.query)
    print("  expected:", query_result.expected)
    print("  got:     ", query_result.retrieved)
```

```
What drove the change in operating margin?
  expected: ['apple_10k_2025.pdf']
  got:      ['<miss>', '<miss>', 'amazon_10k_2025.pdf', '<miss>', '<miss>']
```

`<miss>` marks a retrieved document that was not on the expected list. A result full of misses from the *wrong company* usually means your filters are not being applied. A result full of misses from the *right* document set means your chunking is splitting answers away from the words used to ask about them.

## Matching on something other than file names

By default a retrieved chunk counts as correct when its `source` file name matches an expected entry, ignoring directories and case. To score on a different field, use the mapping form of the golden set:

```yaml title="golden.yaml"
match_field: "company_name"
match_mode: "exact"
queries:
  - query: "What was net income?"
    expected: ["apple"]
```

| `match_mode` | Comparison |
|---|---|
| `basename` | File names only, case insensitive. The default. |
| `exact` | The stored value verbatim, including any path |
| `contains` | The expected value appears anywhere in the stored value |

## Building a golden set that is worth having

- **Twenty real questions beat two hundred synthetic ones.** The value is in encoding judgement the system cannot infer.
- **Include the queries you know are hard.** A golden set of easy questions reports 0.95 recall and teaches you nothing.
- **Write down why a case is there.** The `note` field is ignored by scoring and read by whoever inherits the file.
- **Add a case every time something goes wrong in production.** This is how the set stops being a snapshot and starts being a regression suite.
- **Keep it in version control next to `config.yaml`.** A tuning change with no accompanying eval run is a change nobody can review.

## Using it in CI

`evaluate()` returns plain numbers, so a quality gate is one assertion:

```python
result = evaluate(rag, golden, top_k=5)
assert result.metrics["recall"] >= 0.70, f"recall regressed to {result.metrics['recall']:.3f}"
```

This needs a populated collection, so it belongs in a job that ingests a small fixed corpus first, not in your unit test suite.

## See also

- [Reranking](reranking.md), the change most likely to move MRR
- [Tune Retrieval Quality](tune_retrieval.md) for the settings worth sweeping
