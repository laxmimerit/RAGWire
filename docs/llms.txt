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
pip install "ragwire[google]"
pip install "ragwire[anthropic]"
pip install fastembed        # required for hybrid search
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

llm:
  provider: "ollama"
  model: "qwen3.5:9b"
  base_url: "http://localhost:11434"
  num_ctx: 16384

  # provider: "openai"
  # model: "gpt-4o-mini"

vectorstore:
  url: "http://localhost:6333"
  collection_name: "my_docs"
  use_sparse: true       # hybrid search; requires fastembed
  force_recreate: false

retriever:
  search_type: "hybrid"  # "similarity" | "mmr" | "hybrid"
  top_k: 5
  auto_filter: false
```

---

## Quick Start

```python
from ragwire import RAGWire

rag = RAGWire("config.yaml")

# Ingest (SHA256 deduplication — safe to re-run)
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

## Key Metadata Fields

```python
doc.metadata["company_name"]   # str, lowercased
doc.metadata["fiscal_year"]    # list[int], e.g. [2025] — pass int when filtering
doc.metadata["file_name"]      # str
doc.metadata["source"]         # str, full path
```

---

## Agentic RAG — LangChain Agent

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
