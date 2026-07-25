# Answering Questions

`retrieve()` gives you chunks. `query()` gives you an answer, with the sources it came from.

## The whole thing

```python
from ragwire import RAGWire

rag = RAGWire("config.yaml")
answer = rag.query("What was Apple's net income in fiscal 2025?")

print(answer.text)
```

```
Apple reported net income of $93.7 billion for fiscal 2025 [1], up from
$87.2 billion the prior year [1].
```

No extra configuration. `query()` uses the same LLM already configured for metadata extraction, so if ingestion works, this works.

## Why the numbers in brackets matter

Every claim points at a source. You can resolve them:

```python
for citation in answer.citations:
    print(f"[{citation.index}] {citation.source}")
    print(f"    {citation.snippet}")
```

```
[1] apple_10k_2025.pdf
    Net income for fiscal 2025 was $93,736 million compared to $87,209...
```

Or print the answer with its sources attached:

```python
print(answer.formatted())
```

```
Apple reported net income of $93.7 billion for fiscal 2025 [1].

Sources:
  [1] apple_10k_2025.pdf
```

This is the difference between a demo and something you can put in front of users who will, correctly, want to know where an answer came from.

## Refusing is a feature

The most dangerous RAG failure is not a wrong answer, it is a confident answer assembled from the model's own training data when your documents said nothing at all. `query()` is built to refuse instead:

```python
answer = rag.query("What is the capital of France?")

if answer.refused:
    print("Not in the collection")
else:
    print(answer.text)
```

An `Answer` is falsy when it was refused, so this works too:

```python
if not answer:
    return "I could not find that in the documents."
```

Two things trigger a refusal: retrieval came back empty, or the model reported that the retrieved chunks do not contain the answer. In the first case no LLM call is made at all, since there is nothing to send.

The model is instructed to signal this with a fixed sentinel rather than a phrase, so detecting a refusal never depends on matching prose like "I'm sorry" or "I don't have".

## Reading confidence honestly

```python
print(answer.confidence)   # 0.83
```

`confidence` is the fraction of the answer's sentences that carry a citation.

!!! warning "This is groundedness, not accuracy"
    A fully cited answer drawn from a chunk that happens to be wrong still scores 1.0. What this tells you is how much of the answer is traceable to a source, which is checkable automatically. Whether the source is correct is not.

Used that way it is genuinely useful. A confidence of 0.4 means more than half the answer is unsupported prose the model added on its own, which is worth surfacing or logging even when the answer looks fine.

## Filters work exactly as they do in retrieve()

```python
answer = rag.query(
    "What were the risk factors?",
    filters={"company_name": "nvidia", "fiscal_year": 2025},
)

print(answer.filters_used)   # {'company_name': 'nvidia', 'fiscal_year': 2025}
```

With `auto_filter: true` in your config, filters are extracted from the question automatically and `filters_used` reports what was applied. That matters for debugging: when an answer looks wrong, the first question is usually whether the filters silently excluded the right document.

Filter extraction runs once per `query()` call, not twice, even when extraction finds nothing.

## Async

For anything serving concurrent users, use `aquery()`:

```python
answer = await rag.aquery("What was net income?")
```

Only the LLM call is awaited. Retrieval still runs synchronously, which is the right trade for now: the generation call is the slow part by an order of magnitude, and awaiting it is what keeps a web server's event loop free.

```python title="FastAPI"
from fastapi import FastAPI
from ragwire import RAGWire

app = FastAPI()
rag = RAGWire("config.yaml")

@app.post("/ask")
async def ask(question: str):
    answer = await rag.aquery(question)
    return answer.to_dict()
```

`to_dict()` serialises the answer, its citations and the filters used, which is usually exactly the JSON response you want.

## The context budget

A default RAGWire chunk is 10,000 characters. Five of them is 50,000 characters, which overflows most context windows and, with Ollama's default `num_ctx`, gets silently truncated by the provider, usually dropping the last chunk, which after reranking is not the worst one.

`query()` avoids that by imposing its own budget:

```yaml title="config.yaml"
generation:
  max_context_chars: 12000
```

Chunks are added in rank order until the budget runs out. A chunk that crosses the line is truncated, and one that would land under 200 characters is dropped rather than given a source number it cannot justify. Only chunks that actually fit can be cited, so a citation never points at text the model never saw.

If you raise this, raise `llm.num_ctx` to match. Roughly four characters per token is a safe estimate.

## Changing the instructions

The default prompt tells the model to use only the sources, cite every claim, quote figures exactly, and refuse rather than guess. To change it:

```yaml title="config.yaml"
generation:
  system_prompt: |
    You answer questions about SEC filings for a compliance team.

    Use only the numbered sources. Cite every claim like [2].
    Quote all figures exactly as filed, including units.
    If the sources do not answer the question, reply with exactly {sentinel}.
    Write in complete sentences and never speculate about intent.
```

`{sentinel}` is replaced with the refusal token. Include it, or refusal detection stops working and the model will answer from general knowledge when your documents come up short.

## Composing instead

`query()` is a convenience, not a lock-in. The generator is a separate object that takes documents and returns an answer, so you can drive it yourself:

```python
from ragwire.generation import AnswerGenerator

docs = rag.retrieve("net income", top_k=10)
docs = my_custom_filter(docs)

generator = AnswerGenerator(rag.llm, max_context_chars=20000)
answer = generator.generate("What was net income?", docs)
```

It holds no retrieval logic, so it works with any strategy, including one that is not RAGWire's.

## See also

- [Reranking](reranking.md), which decides which chunks the answer is built from
- [Measure Retrieval Quality](evaluation.md), because a bad answer is usually a retrieval problem
- [Metadata & Filtering](../metadata.md) for scoping a question to the right documents
