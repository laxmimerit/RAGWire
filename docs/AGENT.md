# RAGWire

RAGWire is a Python RAG toolkit for document ingestion and retrieval. Install with `pip install ragwire`. The `RAGWire` class is the single entry point.

- PyPI: https://pypi.org/project/ragwire/
- GitHub: https://github.com/laxmimerit/ragwire
- Docs: https://laxmimerit.github.io/RAGWire/
- Full API reference: https://laxmimerit.github.io/RAGWire/llms-full.txt

---

## Installation

```bash
pip install ragwire          # Ollama embedding + local Qdrant
pip install "ragwire[openai]"
pip install "ragwire[openrouter]"   # LLM + embeddings; Python >= 3.10
pip install "ragwire[google]"
pip install "ragwire[anthropic]"
pip install fastembed        # required for hybrid search

pip install "ragwire[rerank]"  # local cross-encoder reranking, no API key
pip install "ragwire[cohere]"  # hosted reranking
pip install "ragwire[mcp]"     # MCP server for Claude Desktop / Code / Cursor
pip install "ragwire[s3]"      # S3 source connector for rag.sync()
```

---

## Qdrant Setup

**Option 1, local via Docker:**
```bash
docker run -p 6333:6333 qdrant/qdrant
```

**Option 2, Qdrant Cloud:** Create a free cluster at https://cloud.qdrant.io, then set in `config.yaml`:
```yaml
vectorstore:
  url: "https://your-cluster.qdrant.io"
  api_key: "${QDRANT_API_KEY}"
```

**For CLI agents: before writing or executing any RAGWire code, first verify Qdrant is running:**
```bash
curl -sf http://localhost:6333/healthz || { echo "ERROR: Qdrant is not running. Start it with: docker run -p 6333:6333 qdrant/qdrant"; exit 1; }
```

---

## config.yaml

```yaml
embeddings:
  provider: "ollama"
  model: "qwen3-embedding:0.6b"
  base_url: "http://localhost:11434"

  # provider: "openai"
  # model: "text-embedding-3-small"

  # provider: "openrouter"
  # model: "nvidia/llama-nemotron-embed-vl-1b-v2:free"
  # api_key: "${OPENROUTER_API_KEY}"

llm:
  provider: "ollama"
  model: "qwen3.5:9b"
  base_url: "http://localhost:11434"
  num_ctx: 16384

  # provider: "openai"
  # model: "gpt-4o-mini"

  # provider: "openrouter"
  # model: "poolside/laguna-m.1:free"
  # api_key: "${OPENROUTER_API_KEY}"

vectorstore:
  url: "http://localhost:6333"
  collection_name: "my_docs"
  use_sparse: true       # hybrid search; requires fastembed
  force_recreate: false

retriever:
  search_type: "hybrid"  # "similarity" | "mmr" | "hybrid"
  top_k: 5
  auto_filter: false

  # Optional. Omit the block and no reranking happens.
  # rerank:
  #   provider: "cross_encoder"        # or "cohere"
  #   model: "BAAI/bge-reranker-base"
  #   fetch_k: 25                      # default max(4 * top_k, 20)

# Optional. Controls rag.query() only.
generation:
  max_context_chars: 12000

# Optional. Used by rag.sync().
# sources:
#   - type: local
#     path: "./documents"
#     recursive: true
#   - type: s3
#     bucket: "my-filings"
#     prefix: "2026/"
```

---

## Quick Start

```python
from ragwire import RAGWire

rag = RAGWire("config.yaml")

# Ingest. SHA256 deduplication makes this safe to re-run.
stats = rag.ingest_documents(["data/file.pdf"])
stats = rag.ingest_directory("data/", recursive=True)

# Retrieve
results = rag.retrieve("What is the total revenue?", top_k=5)
for doc in results:
    print(doc.page_content)
    print(doc.metadata["file_name"])

# Retrieve with filters
results = rag.retrieve("Net income", filters={"company_name": "apple", "fiscal_year": 2025})

# Agent-controlled filtering (recommended)
filters = rag.extract_filters("Apple's revenue in 2025")
# → {"company_name": "apple", "fiscal_year": 2025} or None
results = rag.retrieve("Apple's revenue in 2025", filters=filters)
```

**Filter rules:**
- String values must be lowercase: `"apple"` not `"Apple"`
- `fiscal_year` takes `int`: `{"fiscal_year": 2025}` not `"2025"`
- List values use OR logic: `{"fiscal_year": [2023, 2024]}` matches either year
- Multiple fields use AND logic

---

## Answering Questions

`rag.query()` returns a grounded answer instead of raw chunks. It cites every claim and refuses rather than answering from general knowledge.

```python
answer = rag.query("What was Apple's net income in fiscal 2025?")

answer.text          # "Apple reported net income of $93.7 billion [1]."
answer.citations     # [Citation([1] apple_10k_2025.pdf)]
answer.sources       # ["apple_10k_2025.pdf"]
answer.filters_used  # filters applied during retrieval, or None
answer.refused       # True when the documents do not answer the question
answer.confidence    # fraction of sentences carrying a citation

print(answer.formatted())   # answer plus a numbered source list
answer.to_dict()            # JSON-ready, for an API response

# An Answer is falsy when refused.
if not answer:
    print("Not in the collection")

# Async: only the LLM call is awaited.
answer = await rag.aquery("What was net income?")
```

Same parameters as `retrieve()`: `top_k`, `filters`, `rerank`.

**`confidence` is groundedness, not accuracy.** It measures how much of the answer is traceable to a source. A fully cited answer built on a wrong chunk still scores 1.0.

---

## Reranking

First-stage retrieval scores query and document separately. A reranker reads the pair together and is far more accurate. Configure `retriever.rerank` and `retrieve()` fetches `fetch_k` candidates, scores them, and returns the best `top_k`.

```python
results = rag.retrieve("operating margin drivers")
results[0].metadata["rerank_score"]   # present only when reranking ran

# Compare against the unreranked baseline
baseline = rag.retrieve("operating margin drivers", rerank=False)
```

`rerank=True` raises `ValueError` if no reranker is configured, rather than silently returning unreranked results. Reranking applies to `retrieve()` and `query()` only; `hybrid_search()` and `mmr_search()` are primitives and ignore it.

---

## Evaluation

```python
from ragwire.eval import GoldenSet, evaluate, sweep

golden = GoldenSet.from_file("golden.yaml")
print(evaluate(rag, golden, top_k=5))     # recall, mrr, hit_rate, precision

print(sweep(rag, golden, {                 # compare settings
    "no rerank": {"rerank": False},
    "reranked":  {"rerank": True},
}))
```

```yaml
# golden.yaml
- query: "What was Apple's net income in fiscal 2025?"
  expected: ["apple_10k_2025.pdf"]
  filters: {company_name: "apple"}   # optional
```

`result.failures` lists the queries that retrieved nothing correct, which is the part worth reading. Unchanged recall with improved MRR is what a working reranker looks like: it reorders the candidate pool rather than widening it.

---

## Syncing Sources

Ingestion only ever adds, so a file deleted at the source keeps answering queries forever. `rag.sync()` reconciles instead.

```python
stats = rag.sync()                      # ingest new, replace changed, delete gone
stats = rag.sync(dry_run=True)          # report without writing or deleting
stats = rag.sync(delete_missing=False)  # additive only
```

**Deletion safety:** if any source fails to list, or lists zero files, sync deletes nothing that run and records why in `stats["warnings"]`. An unreachable bucket and an emptied bucket are indistinguishable from outside. Ingestion still runs.

Custom sources take one class:

```python
from ragwire.sources import REGISTRY, Source

class MySource(Source):
    type_name = "mine"
    def list_files(self):
        return ["/local/cache/a.pdf"]   # raise on failure, never return []

REGISTRY.register(MySource)
```

---

## Command Line

```bash
ragwire ingest ./documents --recursive
ragwire sync --dry-run
ragwire eval golden.yaml --compare-rerank
ragwire mcp serve --config config.yaml
ragwire --version
```

---

## MCP Server

Exposes a collection to Claude Desktop, Claude Code and Cursor.

```bash
pip install "ragwire[mcp]"
ragwire mcp serve --config /absolute/path/to/config.yaml
```

```json
{
  "mcpServers": {
    "ragwire": {
      "command": "ragwire",
      "args": ["mcp", "serve", "--config", "/absolute/path/to/config.yaml"]
    }
  }
}
```

Tools exposed: `get_filter_context`, `search_documents`, `answer_question`, `collection_stats`. Use an absolute config path, since the client launches the server from a working directory you do not control. Paths inside the config, such as `metadata.config_file`, resolve against the config file's own directory and can stay relative.

The same functions work without MCP:

```python
from ragwire.mcp import search_documents, get_filter_context
text = search_documents(rag, "revenue", top_k=5, filters={"company_name": "apple"})
```

---

## Key Metadata Fields

```python
doc.metadata["company_name"]   # str, lowercased
doc.metadata["fiscal_year"]    # int, e.g. 2025
doc.metadata["file_name"]      # str
doc.metadata["source"]         # str, full path
```

---

## Agentic RAG with a LangChain Agent

```python
from typing import Optional
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from ragwire import RAGWire

rag = RAGWire("config.yaml")
rag.ingest_directory("data/")

@tool
def get_filter_context(query: str) -> str:
    """Get available metadata fields and filter suggestions for this query.
    Call before search_documents when the query involves specific metadata
    (company, year, document type). Skip for purely semantic queries."""
    return rag.get_filter_context(query)

@tool
def search_documents(query: str, filters: Optional[dict] = None) -> str:
    """Search the document knowledge base.
    Args:
        query: The search query
        filters: Optional metadata filters from get_filter_context. Pass {} to search without filtering."""
    results = rag.retrieve(query, top_k=5, filters=filters)
    if not results:
        return "No relevant documents found."
    chunks = []
    for doc in results:
        source = doc.metadata.get("file_name", "unknown")
        meta = {k: v for k, v in doc.metadata.items() if k != "file_name" and v not in (None, "", [])}
        chunks.append(f"[{source} | {meta}]\n{doc.page_content}")
    return "\n\n---\n\n".join(chunks)

model = ChatOllama(model="qwen3.5:9b", base_url="http://localhost:11434")
agent = create_agent(
    model=model,
    tools=[get_filter_context, search_documents],
    system_prompt=(
        "Always use search_documents to retrieve information. "
        "Use get_filter_context before search_documents when the query involves "
        "specific metadata (company, year, document type). "
        "Never answer from general knowledge. Always cite the source."
    ),
    checkpointer=InMemorySaver(),
)

response = agent.invoke(
    {"messages": [{"role": "user", "content": "What is Apple's 2025 revenue?"}]},
    config={"configurable": {"thread_id": "session-1"}},
)
print(response["messages"][-1].content)
```

**Agent reasoning flow:**
1. Query arrives
2. Call `get_filter_context` if query mentions specific company/year/type
3. Use returned context to decide `filters` dict
4. Call `search_documents(query, filters=filters)`
5. Answer from retrieved chunks

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| Qdrant connection refused | `docker run -p 6333:6333 qdrant/qdrant` |
| `fastembed` missing | `pip install fastembed` |
| Ollama model not found | `ollama pull <model-name>` |
| Embedding dimension mismatch | Set `force_recreate: true` once, then back to `false` |
| Filter returns no results | Check values with `rag.get_field_values("company_name")` |
