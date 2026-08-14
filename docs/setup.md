# Installation and Setup

This page takes you from nothing to a working RAG pipeline. Every step is
copy-pasteable, and the whole thing runs locally with no API key.

If you would rather follow a narrative walkthrough that ends in a working
script, use the [Quick Tutorial](tutorial.md) instead. This page is the
reference version.

## What you need

| Requirement | Why | Check it works |
|---|---|---|
| Python 3.10+ | RAGWire requires it | `python --version` |
| [Docker](https://www.docker.com/) | Runs the Qdrant vector database | `docker --version` |
| [Ollama](https://ollama.com/) | Runs the embedding and chat models locally | `ollama --version` |

Ollama is only needed for the local, no-API-key setup shown here. If you would
rather use a hosted provider, swap the `embeddings` and `llm` blocks for one of
the [provider configs](openai.md) and skip step 3.

## Step 1: Install RAGWire

```bash
pip install "ragwire[ollama]" fastembed
```

`fastembed` powers the sparse half of hybrid search. Without it RAGWire still
works, but falls back to dense-only search.

Other providers install the same way:

```bash
pip install "ragwire[openai]"       # OpenAI
pip install "ragwire[openrouter]"   # OpenRouter, LLM and embeddings
pip install "ragwire[google]"       # Google Gemini
pip install "ragwire[huggingface]"  # HuggingFace, local
pip install "ragwire[groq]"         # Groq
pip install "ragwire[anthropic]"    # Anthropic Claude
pip install "ragwire[all]"          # Everything
```

## Step 2: Start Qdrant

Qdrant is the vector database where your chunks are stored.

```bash
docker run -d -p 6333:6333 --name qdrant qdrant/qdrant
```

Verify it is up:

```bash
curl http://localhost:6333/healthz
```

You want `{"title":"qdrant - vector search engine"}`. If you get a connection
error, Docker is not running or the container did not start; check with
`docker ps`.

## Step 3: Pull the Ollama models

RAGWire uses two models: one to embed chunks, one to read each document and
extract its metadata.

```bash
ollama pull nomic-embed-text
ollama pull qwen3.5:9b
```

Confirm both are present:

```bash
ollama list
```

The chat model is the slow part of ingestion. On a machine without a GPU,
substitute something smaller such as `qwen3.5:2b` and update `llm.model` below
to match.

## Step 4: Create config.yaml

Put this in your project directory. It is the complete working configuration
for the setup above.

```yaml
# Which file types to pick up when scanning a directory
loader:
  extensions: [".pdf", ".docx", ".xlsx", ".pptx", ".txt", ".md"]

# How documents are cut into chunks
splitter:
  chunk_size: 10000
  chunk_overlap: 2000
  strategy: "markdown"   # "markdown", "recursive" or "page"
  # "page" stores exactly one chunk per PDF page / PPTX slide and adds
  # page_number to every chunk; chunk_size and chunk_overlap then no longer
  # apply. Text files split on page_marker (default "<!-- pagebreak -->").

# Turns chunks into vectors
embeddings:
  provider: "ollama"
  model: "nomic-embed-text"
  base_url: "http://localhost:11434"

# Reads each document and extracts metadata. Required.
llm:
  provider: "ollama"
  model: "qwen3.5:9b"
  base_url: "http://localhost:11434"
  num_ctx: 65536         # Optional. Raise if your chunks are large.

# Where the vectors live
vectorstore:
  url: "http://localhost:6333"
  collection_name: "my_docs"
  use_sparse: true       # Hybrid search. Needs fastembed.
  force_recreate: false  # true wipes the collection on startup. Leave false.

# How documents come back out
retriever:
  search_type: "hybrid"  # "similarity", "mmr" or "hybrid"
  top_k: 5
  auto_filter: false     # true lets the LLM derive metadata filters per query

logging:
  level: "INFO"
  console_output: true
  colored: true
```

Two settings are worth understanding before you run anything:

- **`force_recreate`** deletes the entire collection every time your program
  starts. Keep it `false` unless you are deliberately rebuilding.
- **`llm` is required.** RAGWire extracts metadata from every document at
  ingest time, which is what makes filtered retrieval work. There is no
  metadata-free mode.

## Step 5: Run it

```python
from ragwire import RAGWire

rag = RAGWire("config.yaml")

stats = rag.ingest_documents(["data/report.pdf"])
print(stats)

for doc in rag.retrieve("What was the total revenue?"):
    print(doc.metadata.get("file_name"), doc.page_content[:200])
```

A healthy first run prints something like:

```python
{'total': 1, 'processed': 1, 'skipped': 0, 'failed': 0,
 'chunks_created': 42, 'metadata_failed': 0, 'replaced': 0, 'errors': []}
```

Those counters always reconcile: `processed + skipped + failed == total`. If a
document is missing from your results, this is the first place to look.

## Tuning ingestion

Everything here is optional and the defaults are deliberately conservative.

```yaml
ingestion:
  workers: 1            # Documents prepared in parallel
  batch_size: 64        # Chunks per write request
  retries: 2            # Retries per write, with exponential backoff
  replace_changed: true # Replace a document when its content changes
  dedup_chunks: false   # Drop repeated chunks within one document
```

- **`workers`** is the one to raise for large batches. Loading, splitting and
  the metadata call dominate ingest time, and those run in parallel; writes stay
  sequential. Leave it at `1` when using local file storage, which is
  single-process only.
- **`replace_changed`** means editing a file and re-ingesting replaces the old
  version instead of storing both. Turn it off only if you want every version
  retained.
- **`dedup_chunks`** helps with long filings that repeat the same boilerplate
  across sections and crowd out real content in results.

## Using environment variables

Any value in `config.yaml` can reference an environment variable with `${VAR}`:

```yaml
vectorstore:
  url: "https://your-cluster.qdrant.io"
  api_key: "${QDRANT_API_KEY}"

llm:
  api_key: "${OPENAI_API_KEY}"
```

RAGWire resolves these at startup via `python-dotenv`, reading a `.env` file in
your working directory. If a variable is not set, the placeholder is left as-is
and a warning is logged, which usually surfaces later as an authentication
error from the provider.

## When something goes wrong

| What you see | What it means |
|---|---|
| `Missing [embeddings] section` or `Missing [vectorstore] section` | `config.yaml` was found but is incomplete. The message names the missing block. |
| `Configuration file not found` | The path passed to `RAGWire(...)` is wrong, or you are running from a different directory. |
| Connection refused on port 6333 | Qdrant is not running. See step 2. |
| Connection refused on port 11434 | Ollama is not running. Start it and re-check `ollama list`. |
| `Embedding dimension mismatch` | You changed `embeddings.model` after ingesting. Either change it back, or set `force_recreate: true` once and re-ingest. The message names both dimensions. |
| `failed: 1` with "no extractable text" | A scanned or image-only PDF with no text layer. RAGWire does not OCR; run the file through an OCR tool first. |
| `metadata_failed` above zero | Documents were ingested but the LLM was unreachable, so they carry no metadata and will not match filters. Fix the LLM, then `rag.reingest_documents([...])`. |
| Filters return nothing on local storage | Local file storage cannot use payload indexes. RAGWire falls back to scanning points, which is exact but slower. Use a Qdrant server for large collections. |

## LangSmith tracing (optional)

To trace LLM calls via [LangSmith](https://smith.langchain.com), add these to
`.env`:

```env
LANGSMITH_API_KEY=your_api_key_here
LANGSMITH_PROJECT=RAGWire
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_TRACING_V2=true
```

No code changes are needed. RAGWire uses LangChain internally, so every LLM
call is traced automatically. See [LangSmith Tracing](cookbook/langsmith.md).

## Next steps

- [Quick Tutorial](tutorial.md) builds a complete pipeline end to end.
- [Metadata and Filtering](metadata.md) covers filtered retrieval.
- [Custom Metadata](custom_metadata.md) replaces the built-in financial fields
  with your own schema.
