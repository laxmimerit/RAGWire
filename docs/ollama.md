# RAGWire with Ollama

Ollama lets you run LLMs and embedding models locally — no API key, no cost, no data leaving your machine.

## Prerequisites

- [Ollama](https://ollama.com/) installed and running
- RAGWire installed: `pip install "ragwire[ollama]"`
- Qdrant running: `docker run -d -p 6333:6333 qdrant/qdrant`

## 1. Pull Models

Pull an embedding model and a chat model:

```bash
# Embedding model (small, fast)
ollama pull qwen3-embedding:0.6b

# Chat model for metadata extraction
ollama pull qwen3.5:9b
```

Verify they are available:

```bash
ollama list
```

Other good options:

```bash
# Embeddings
ollama pull nomic-embed-text        # 274MB, fast, very popular
ollama pull mxbai-embed-large       # Higher quality, BERT-large class
ollama pull bge-m3                  # Multilingual, multi-granularity
ollama pull qwen3-embedding:4b      # 2.5GB, higher quality
ollama pull qwen3-embedding:8b      # 4.7GB, best quality, 40K context

# Chat
ollama pull llama3.1                # Meta, 8B/70B sizes — most popular
ollama pull qwen3                   # Alibaba, dense + MoE variants
ollama pull deepseek-r1             # Reasoning model
ollama pull gemma3                  # Google, runs on single GPU
```

## 2. Install Dependencies

```bash
pip install "ragwire[ollama]"
pip install fastembed               # For hybrid search
```

## 3. Configuration

```yaml
embeddings:
  provider: "ollama"
  model: "qwen3-embedding:0.6b"
  base_url: "http://localhost:11434"

llm:
  provider: "ollama"
  model: "qwen3.5:9b"
  base_url: "http://localhost:11434"
  num_ctx: 16384

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

## 6. Build a RAG Agent

Use `create_agent` to wrap the retriever as a tool and build a conversational Q&A app:

```python
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama
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
    model=ChatOllama(model="qwen3.5:9b"),
    tools=[search_documents],
    system_prompt=(
        "You are a helpful document assistant. "
        "Always use search_documents to retrieve information before answering — never answer from general knowledge. "
        "If no relevant documents are found, say so — do not guess or fabricate an answer. "
        "Always cite the source document in your answer."
    ),
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

- `num_ctx` must be large enough to fit your chunk size. Default `10000` chars → set `num_ctx: 16384` or higher.
- Ollama runs on `http://localhost:11434` by default. If you changed the port, update `base_url`.
- For GPU acceleration, Ollama detects CUDA/Metal automatically — no extra config needed.
