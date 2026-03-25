# Quick Tutorial

This tutorial walks you through a complete RAG pipeline from scratch using RAGWire and Ollama (fully local, no API key required). By the end you will have ingested a PDF and retrieved relevant chunks for any query.

---

## Prerequisites

- Python 3.9+
- [Docker](https://www.docker.com/) installed
- [Ollama](https://ollama.com/) installed

---

## Step 1 — Install RAGWire

```bash
pip install "ragwire[ollama]"
pip install fastembed
```

`fastembed` is needed for hybrid search (dense + sparse vectors).

---

## Step 2 — Start Qdrant

Qdrant is the vector database where your document chunks are stored.

```bash
docker run -d -p 6333:6333 --name qdrant qdrant/qdrant
```

Verify it is running:

```bash
curl http://localhost:6333/healthz
```

You should see:

```json
{"title":"qdrant - vector search engine"}
```

---

## Step 3 — Pull Ollama Models

You need two models — one for embeddings, one for metadata extraction.

```bash
ollama pull nomic-embed-text       # Embedding model (274MB)
ollama pull qwen3.5:9b             # Chat model for metadata extraction
```

Verify both are available:

```bash
ollama list
```

---

## Step 4 — Create Configuration

Create a `config.yaml` file in your project directory:

```yaml
loader:
  extensions: [".pdf", ".docx", ".xlsx", ".pptx", ".txt", ".md"]

splitter:
  chunk_size: 10000
  chunk_overlap: 2000
  strategy: "markdown"

embeddings:
  provider: "ollama"
  model: "nomic-embed-text"
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
  auto_filter: false   # set true to enable LLM-based filter extraction from every query

logging:
  level: "INFO"
  console_output: true
  colored: true
```

---

## Step 5 — Add Documents

Create a `data/` folder and place your PDF (or DOCX, XLSX, etc.) inside:

```
your-project/
├── config.yaml
└── data/
    └── Apple_10k_2025.pdf
```

---

## Step 6 — Run the Pipeline

Create a Python script `run.py`:

```python
from ragwire import RAGWire

# Initialize — loads config, connects to Qdrant, initializes models
rag = RAGWire("config.yaml")

# Ingest documents
stats = rag.ingest_documents(["data/Apple_10k_2025.pdf"])

print(f"Processed : {stats['processed']}/{stats['total']}")
print(f"Skipped   : {stats['skipped']} (already ingested)")
print(f"Chunks    : {stats['chunks_created']}")
```

Run it:

```bash
python run.py
```

Expected output:

```
INFO - Created new collection: my_docs
INFO - Processed data/Apple_10k_2025.pdf: 42 chunks
Processed : 1/1
Skipped   : 0 (already ingested)
Chunks    : 42
```

!!! tip "Deduplication"
    Run the script a second time — the file will be skipped automatically because RAGWire checks the SHA256 hash before ingesting.

---

## Step 7 — Retrieve Documents

Add retrieval to your script:

```python
from ragwire import RAGWire

rag = RAGWire("config.yaml")

queries = [
    "What is the total revenue?",
    "What are the main product categories?",
    "What are the key risk factors?",
]

for query in queries:
    print(f"\nQuery: {query}")
    print("-" * 50)

    results = rag.retrieve(query, top_k=3)

    for i, doc in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"  Company  : {doc.metadata.get('company_name', 'Unknown')}")
        print(f"  Doc Type : {doc.metadata.get('doc_type', 'Unknown')}")
        print(f"  Year     : {doc.metadata.get('fiscal_year', 'Unknown')}")
        print(f"  Content  : {doc.page_content[:300]}...")
```

---

## Step 8 — Explore Component Usage

You can also use individual components directly:

```python
from ragwire import (
    MarkItDownLoader,
    get_markdown_splitter,
    get_splitter,
    get_embedding,
    QdrantStore,
    MetadataExtractor,
)

# Load a document
loader = MarkItDownLoader()
result = loader.load("data/Apple_10k_2025.pdf")
print(f"Loaded: {result['file_name']}, chars: {len(result['text_content'])}")

# Split into chunks
splitter = get_markdown_splitter(chunk_size=10000, chunk_overlap=2000)
chunks = splitter.split_text(result["text_content"])
print(f"Chunks: {len(chunks)}")

# Create embeddings
embedding = get_embedding({
    "provider": "ollama",
    "model": "nomic-embed-text",
    "base_url": "http://localhost:11434",
})
vector = embedding.embed_query("test query")
print(f"Embedding dimension: {len(vector)}")

# Connect to vector store
store = QdrantStore(
    config={"url": "http://localhost:6333"},
    embedding=embedding,
    collection_name="my_docs",
)
vectorstore = store.get_store(use_sparse=True)
results = vectorstore.similarity_search("total revenue", k=3)
print(f"Retrieved: {len(results)} chunks")
```

---

## Step 9 — Hybrid Search

Hybrid search combines dense (semantic) and sparse (keyword) vectors for better recall. It is enabled by default when `use_sparse: true` and `search_type: "hybrid"` are set in `config.yaml`.

You can also call it directly:

```python
from ragwire import RAGWire

rag = RAGWire("config.yaml")

# Hybrid search (dense + sparse)
results = rag.hybrid_search("Apple total revenue fiscal 2025", k=5)

for doc in results:
    print(doc.page_content[:200])
```

---

## Step 10 — Switch Providers

Switching providers only requires changing `config.yaml` — no code changes needed.

=== "Ollama (local)"

    ```yaml
    embeddings:
      provider: "ollama"
      model: "nomic-embed-text"

    llm:
      provider: "ollama"
      model: "qwen3.5:9b"
      num_ctx: 16384
    ```

=== "OpenAI"

    ```yaml
    embeddings:
      provider: "openai"
      model: "text-embedding-3-small"

    llm:
      provider: "openai"
      model: "gpt-5.4-nano"
    ```

=== "Gemini"

    ```yaml
    embeddings:
      provider: "google"
      model: "models/gemini-embedding-001"

    llm:
      provider: "google"
      model: "gemini-2.5-flash"
    ```

=== "Groq + Ollama"

    ```yaml
    embeddings:
      provider: "ollama"
      model: "nomic-embed-text"

    llm:
      provider: "groq"
      model: "llama-3.3-70b-versatile"
    ```

=== "Anthropic + Ollama"

    ```yaml
    embeddings:
      provider: "ollama"
      model: "nomic-embed-text"

    llm:
      provider: "anthropic"
      model: "claude-haiku-4-5-20251001"
    ```

!!! warning "Switching embedding models"
    If you change the embedding model, you must set `force_recreate: true` in `vectorstore` once to rebuild the collection — then set it back to `false`.

---

## Troubleshooting

| Error | Fix |
|---|---|
| `Qdrant connection refused` | Run `docker run -p 6333:6333 qdrant/qdrant` |
| `Ollama model not found` | Run `ollama pull <model-name>` |
| `fastembed` missing | `pip install fastembed` |
| `markitdown[pdf]` missing | `pip install "markitdown[pdf]"` |
| Embedding dimension mismatch | Set `force_recreate: true` once, then back to `false` |
| Collection has no sparse vectors | Set `force_recreate: true` once, then back to `false` |
