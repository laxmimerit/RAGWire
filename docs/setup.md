# Installation & Setup

## Prerequisites

- Python 3.9 or higher
- [Docker](https://www.docker.com/) (for running Qdrant locally)

## Install RAGWire

Install the base package:

```bash
pip install ragwire
```

Install with a specific provider:

```bash
pip install "ragwire[ollama]"       # Ollama (local, no API key)
pip install "ragwire[openai]"       # OpenAI
pip install "ragwire[google]"       # Google Gemini
pip install "ragwire[huggingface]"  # HuggingFace (local)
pip install "ragwire[groq]"         # Groq
pip install "ragwire[anthropic]"    # Anthropic Claude
pip install "ragwire[all]"          # Everything
```

For hybrid search (dense + sparse), also install:

```bash
pip install fastembed
```

## Start Qdrant

RAGWire uses [Qdrant](https://qdrant.tech/) as its vector database. Run it locally with Docker:

```bash
docker run -d -p 6333:6333 --name qdrant qdrant/qdrant
```

Verify it is running:

```bash
curl http://localhost:6333/healthz
# {"title":"qdrant - vector search engine"}
```

## Create config.yaml

Create a `config.yaml` file in your project directory. Below is a full working example using Ollama (local, no API key):

```yaml
# =================================================================
# Document Loader
# =================================================================
loader:
  extensions: [".pdf", ".docx", ".xlsx", ".pptx", ".txt", ".md"]

# =================================================================
# Text Splitter
# =================================================================
splitter:
  chunk_size: 10000
  chunk_overlap: 2000
  strategy: "markdown"   # "markdown" | "recursive"

# =================================================================
# Embedding Model
# =================================================================
embeddings:
  provider: "ollama"
  model: "nomic-embed-text"
  base_url: "http://localhost:11434"

# =================================================================
# LLM — used for metadata extraction on every ingested document
# =================================================================
llm:
  provider: "ollama"
  model: "qwen3.5:9b"
  base_url: "http://localhost:11434"
  num_ctx: 65536         # optional — override context window if chunks are large

# =================================================================
# Metadata Extraction (optional)
# =================================================================
# metadata:
#   config_file: "metadata.yaml"   # Custom fields (see Custom Metadata docs)
#                                  # If omitted, defaults to: company_name,
#                                  # doc_type, fiscal_quarter, fiscal_year

# =================================================================
# Vector Store (Qdrant)
# =================================================================
vectorstore:
  url: "http://localhost:6333"
  collection_name: "my_docs"
  use_sparse: true       # Enables hybrid search (requires: pip install fastembed)
  force_recreate: false  # Set true ONLY to wipe and rebuild the collection

# =================================================================
# Retriever
# =================================================================
retriever:
  search_type: "hybrid"  # "similarity" | "mmr" | "hybrid"
  top_k: 5
  auto_filter: false     # set true to enable LLM-based filter extraction from every query

# =================================================================
# Logging
# =================================================================
logging:
  level: "INFO"
  console_output: true
  colored: true
  # log_file: "logs/rag.log"
```

See the provider docs for equivalent configs using OpenAI, Google, Groq, or HuggingFace.

### Using environment variables

Any value in `config.yaml` can reference an environment variable using `${VAR}` syntax:

```yaml
vectorstore:
  url: "https://your-cluster.qdrant.io"
  api_key: "${QDRANT_API_KEY}"

llm:
  api_key: "${OPENAI_API_KEY}"
```

RAGWire resolves these at startup via `python-dotenv`. If a variable is not set, the placeholder is kept and a warning is logged.

### LangSmith tracing (optional)

To enable LLM call tracing via [LangSmith](https://smith.langchain.com), add these to your `.env` file:

```env
LANGSMITH_API_KEY=your_api_key_here
LANGSMITH_PROJECT=RAGWire
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_TRACING_V2=true
```

No code changes needed — RAGWire uses LangChain internally so all LLM calls are traced automatically. See [LangSmith Tracing](cookbook/langsmith.md) for details.
