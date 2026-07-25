# RAGWire with OpenRouter

Use [OpenRouter](https://openrouter.ai) for both embeddings and the metadata extraction LLM. OpenRouter gives you a single API key and access to models from many providers, including free-tier models.

RAGWire uses the dedicated **`ChatOpenRouter`** integration for the LLM (so structured metadata extraction works reliably) and the official **`openrouter` Python SDK** for embeddings.

## Prerequisites

- OpenRouter API key from [openrouter.ai/settings/keys](https://openrouter.ai/settings/keys)
- **Python 3.10 or higher**, required by `langchain-openrouter`
- RAGWire installed: `pip install "ragwire[openrouter]"`
- Qdrant running: `docker run -d -p 6333:6333 qdrant/qdrant`

## 1. Install Dependencies

```bash
pip install "ragwire[openrouter]"   # installs langchain-openrouter + openrouter SDK
pip install fastembed                # For hybrid search (optional)
```

!!! warning "Python version"
    `langchain-openrouter` requires Python **≥ 3.10**. On Python 3.9, use a different provider for the LLM (e.g. Ollama or OpenAI).

## 2. Set API Key

```bash
# Linux / macOS
export OPENROUTER_API_KEY="sk-or-v1-..."

# Windows (PowerShell)
$env:OPENROUTER_API_KEY="sk-or-v1-..."
```

Or add it to a `.env` file at the project root:

```
OPENROUTER_API_KEY=sk-or-v1-...
```

`ChatOpenRouter` and the embeddings client both read `OPENROUTER_API_KEY` automatically. You can also pass it explicitly in config via `api_key: "${OPENROUTER_API_KEY}"`.

## 3. Configuration

```yaml
embeddings:
  provider: "openrouter"
  model: "nvidia/llama-nemotron-embed-vl-1b-v2:free"   # 2048-dim, free
  api_key: "${OPENROUTER_API_KEY}"
  # batch_size: 16        # optional: inputs per request (default 100)
  # dimensions: 2048      # optional: only if the model supports it

llm:
  provider: "openrouter"
  model: "poolside/laguna-m.1:free"                    # free; or any OpenRouter model id
  api_key: "${OPENROUTER_API_KEY}"

vectorstore:
  url: "http://localhost:6333"
  collection_name: "my_docs"
  use_sparse: true
  force_recreate: false

retriever:
  search_type: "hybrid"
  top_k: 5
  auto_filter: false   # set true to enable LLM-based filter extraction from every query
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

## Model Notes

| Type | Example model | Dimensions | Notes |
|---|---|---|---|
| Embeddings | `nvidia/llama-nemotron-embed-vl-1b-v2:free` | 2048 | Free tier |
| Embeddings | `openai/text-embedding-3-small` | 1536 | Paid, via OpenRouter |
| LLM | `poolside/laguna-m.1:free` | n/a | Free tier |
| LLM | `anthropic/claude-sonnet-4.5` | n/a | Paid, via OpenRouter |

Browse all models at [openrouter.ai/models](https://openrouter.ai/models). Filter embedding models with `?fmt=cards&output_modalities=embeddings`.

## 6. Build a RAG Agent

Use `create_agent` to wrap the retriever as a tool and build a conversational Q&A app:

```python
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openrouter import ChatOpenRouter
from langgraph.checkpoint.memory import InMemorySaver
from ragwire import RAGWire

rag = RAGWire("config.yaml")
rag.ingest_directory("data/")

@tool
def search_documents(query: str) -> str:
    """Search the document knowledge base for relevant information."""
    results = rag.retrieve(query, top_k=5)
    if not results:
        return "No relevant documents found."
    return "\n\n---\n\n".join(
        f"[{doc.metadata.get('file_name')}]\n{doc.page_content}"
        for doc in results
    )

agent = create_agent(
    model=ChatOpenRouter(model="poolside/laguna-m.1:free"),
    tools=[search_documents],
    system_prompt=(
        "You are a helpful document assistant. "
        "Always use search_documents to retrieve information before answering. "
        "Never answer from general knowledge. "
        "If no relevant documents are found, say so. Do not guess or fabricate an answer. "
        "Always cite the source document in your answer."
    ),
    checkpointer=InMemorySaver(),
)

config = {"configurable": {"thread_id": "session-1"}}
response = agent.invoke(
    {"messages": [HumanMessage("What is the total revenue?")]},
    config=config,
)
print(response["messages"][-1].content)
```

See [RAG Agent](rag_agent.md) for the full guide including multi-turn memory and structured output.

---

## Notes

- **Mixing providers is fine.** You can use OpenRouter for the LLM and Ollama for embeddings, or the reverse.
- If you change embedding model after ingestion, set `force_recreate: true` once to rebuild the collection, since the dimensions differ (the `nvidia/...` model is 2048-dim).
- The API key can also be passed directly in config as `api_key: "sk-or-v1-..."`, but environment variables are preferred.
- Embeddings go through the official `openrouter` SDK with `encoding_format="float"`. RAGWire does **not** route OpenRouter embeddings through the OpenAI client (which mis-parses OpenRouter's base64 responses).
