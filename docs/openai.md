# RAGWire with OpenAI

Use OpenAI for both embeddings and the metadata extraction LLM.

## Prerequisites

- OpenAI API key — [platform.openai.com](https://platform.openai.com)
- RAGWire installed: `pip install "ragwire[openai]"`
- Qdrant running: `docker run -d -p 6333:6333 qdrant/qdrant`

## 1. Install Dependencies

```bash
pip install "ragwire[openai]"
pip install fastembed               # For hybrid search
```

## 2. Set API Key

```bash
# Linux / macOS
export OPENAI_API_KEY="sk-..."

# Windows (PowerShell)
$env:OPENAI_API_KEY="sk-..."
```

Or add it to a `.env` file at the project root:

```
OPENAI_API_KEY=sk-...
```

## 3. Configuration

```yaml
embeddings:
  provider: "openai"
  model: "text-embedding-3-small"   # 1536-dim, best price/performance
  # model: "text-embedding-3-large" # 3072-dim, highest quality

llm:
  provider: "openai"
  model: "gpt-5.4-nano"             # Latest — fast, affordable, good for metadata extraction
  # model: "gpt-4o-mini"            # Previous generation
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

| Model | Dimensions | Notes |
|---|---|---|
| `text-embedding-3-small` | 1536 | Best price/performance — recommended |
| `text-embedding-3-large` | 3072 | Highest quality, multilingual |
| `text-embedding-ada-002` | 1536 | Legacy — avoid for new projects |

## 6. Build a RAG Agent

Use `create_agent` to wrap the retriever as a tool and build a conversational Q&A app:

```python
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
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
    model=ChatOpenAI(model="gpt-5.4-nano", temperature=0),
    tools=[search_documents],
    system_prompt="You are a helpful document assistant. Use search_documents to answer questions.",
    checkpointer=InMemorySaver(),
)

config = {"configurable": {"thread_id": "session-1"}}
response = agent.invoke(
    {"messages": [{"role": "user", "content": "What is the total revenue?"}]},
    config=config,
)
print(response["messages"][-1].content)
```

See [RAG Agent](rag_agent.md) for the full guide including multi-turn memory and structured output.

---

## Notes

- If you change embedding model after ingestion, set `force_recreate: true` once to rebuild the collection (dimensions will differ).
- The API key can also be passed directly in config: `api_key: "sk-..."` — but environment variables are preferred.
