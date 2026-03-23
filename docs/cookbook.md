# Cookbook

Short, focused recipes for common real-world scenarios. Each recipe is self-contained — copy, adapt, run.

---

## Switch from OpenAI to Local Ollama

Swap your embedding and LLM to run fully offline with no API costs. Only `config.yaml` changes — your Python code stays the same.

```yaml
embeddings:
  provider: "ollama"
  model: "nomic-embed-text"
  base_url: "http://localhost:11434"

llm:
  provider: "ollama"
  model: "qwen3.5:9b"
  base_url: "http://localhost:11434"
  num_ctx: 16384
```

!!! warning "Recreate the collection when switching embedding models"
    Different embedding models produce different vector dimensions. If you already have a collection, set `force_recreate: true` once, run ingestion, then set it back to `false`.

---

## Use Qdrant Without Docker (Local File Storage)

No Docker needed — Qdrant can store vectors in a local folder.

```yaml
vectorstore:
  url: "./qdrant_storage"   # path instead of http://
  collection_name: "my_docs"
  use_sparse: false          # fastembed sparse index not supported in file mode
```

```python
rag = RAGWire("config.yaml")
stats = rag.ingest_directory("data/")
```

The `qdrant_storage/` folder is created automatically. Ideal for local development or single-machine deployments.

---

## Separate Collections per Client or Domain

Use one `RAGWire` instance per collection — useful when serving multiple clients or keeping domains isolated.

```python
from ragwire import RAGWire
import yaml, tempfile, os

def make_rag(collection_name: str) -> RAGWire:
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    config["vectorstore"]["collection_name"] = collection_name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.dump(config, tmp)
        tmp_path = tmp.name
    try:
        return RAGWire(tmp_path)
    finally:
        os.unlink(tmp_path)

rag_legal   = make_rag("legal_docs")
rag_finance = make_rag("financial_docs")
rag_hr      = make_rag("hr_docs")

rag_legal.ingest_directory("data/legal/")
rag_finance.ingest_directory("data/finance/")
```

---

## Update a Document Without Duplicates

RAGWire deduplicates by SHA256 file hash — re-ingesting the same file is a no-op. To update a document, the file content must change (even a single byte triggers re-ingestion).

```python
# First ingest
stats = rag.ingest_documents(["reports/Q1_2025.pdf"])
print(stats["processed"])   # → 1

# Re-run with same file — skipped automatically
stats = rag.ingest_documents(["reports/Q1_2025.pdf"])
print(stats["skipped"])     # → 1

# Update the file, re-run — new version ingested
# (old chunks remain; to remove them, set force_recreate: true and re-ingest all)
stats = rag.ingest_documents(["reports/Q1_2025.pdf"])
print(stats["processed"])   # → 1
```

!!! note "Full replacement"
    RAGWire does not delete old chunks when a file is updated. For a full replacement, set `force_recreate: true` in config and re-ingest all documents.

---

## Tune Retrieval Quality

Three knobs affect what comes back from retrieval:

| Setting | Default | Effect |
|---|---|---|
| `chunk_size` | `10000` | Larger = more context per chunk; smaller = more precise matches |
| `chunk_overlap` | `2000` | Prevents context being cut at chunk boundaries. ~20% of chunk_size is a good default |
| `top_k` | `5` | More results = broader coverage; fewer = higher precision |

```yaml
splitter:
  chunk_size: 5000      # halve chunk size for more precise retrieval
  chunk_overlap: 1000
  strategy: "markdown"

retriever:
  top_k: 8             # return more candidates
  search_type: "hybrid"
```

**Rule of thumb:**
- Long-form documents (10-Ks, contracts) → larger chunks (`8k–12k`)
- FAQ-style or structured content → smaller chunks (`500–2k`)
- Queries needing breadth (summaries) → higher `top_k`
- Queries needing precision (exact figures) → lower `top_k` + explicit filters

---

## Inspect What's in Your Collection

Before querying, understand what's actually stored.

```python
from ragwire import RAGWire

rag = RAGWire("config.yaml")

# Collection size and vector dimension
stats = rag.get_stats()
print(f"Collection : {stats['collection_name']}")
print(f"Total chunks: {stats['total_documents']}")
print(f"Vector size : {stats['vector_size']}")

# What metadata fields exist?
fields = rag.discover_metadata_fields()
print(f"Fields: {fields}")
# → ['company_name', 'doc_type', 'fiscal_year', 'file_name', ...]

# What values are stored for the key fields?
values = rag.get_field_values(["company_name", "doc_type", "fiscal_year"])
print(values)
# → {
#     'company_name': ['apple', 'microsoft', 'google'],
#     'doc_type':     ['10-k', '10-q'],
#     'fiscal_year':  ['2024', '2025'],
# }
```

---

## Get Diverse Results with MMR

By default, `retrieve()` returns the top-N most similar chunks — which can be nearly identical if they come from the same page. Use MMR to get results spread across different parts of the document.

```python
from ragwire import mmr_search

results = mmr_search(
    rag.vectorstore,
    query="Apple revenue breakdown",
    k=5,
    fetch_k=20,      # fetch 20 candidates, pick the 5 most diverse
    lambda_mult=0.3, # lean towards diversity (0.0 = max diverse, 1.0 = max relevant)
)

for doc in results:
    print(doc.metadata.get("chunk_index"), doc.page_content[:100])
```

**When to use MMR:** when your document corpus has long repetitive sections (e.g. boilerplate in legal contracts, repeated tables in financial reports) and similarity search keeps returning the same content.

---

## Load and Inspect Documents Before Ingesting

Preview extracted text before committing to the vector store — useful for debugging loader issues or verifying OCR quality.

```python
from ragwire import MarkItDownLoader

loader = MarkItDownLoader()

# Single file
result = loader.load("data/Apple_10k_2025.pdf")
if result["success"]:
    print(result["text_content"][:500])
else:
    print(f"Error: {result['error']}")

# Whole directory
results = loader.load_directory("data/", extensions=[".pdf", ".docx"])
for r in results:
    status = "OK" if r["success"] else f"FAILED: {r['error']}"
    print(f"{r['file_name']}: {status}")
```

---

## Write Logs to a File

Useful in production or scheduled jobs where you want a persistent audit trail.

```yaml
logging:
  level: "INFO"
  colored: false
  console_output: true
  log_file: "logs/rag.log"
```

Or in code:

```python
from ragwire import setup_logging

logger = setup_logging(log_level="INFO", log_file="logs/rag.log")
```

For development with colored output:

```python
from ragwire import setup_colored_logging

logger = setup_colored_logging(log_level="DEBUG")
```

---

## Use a Custom Metadata Schema for Non-Financial Docs

The default schema extracts financial fields (`company_name`, `doc_type`, `fiscal_year`). For any other domain, define your own fields in a YAML file.

```yaml
# metadata.yaml
fields:
  - name: organization
    description: "Organization name in lowercase"

  - name: doc_type
    description: "Type of document"
    values: ["contract", "policy", "report", "memo", "other"]

  - name: effective_year
    description: "Year the document is effective, as integer or null"

  - name: jurisdiction
    description: "Country or region the document applies to, or null"
```

```yaml
# config.yaml
metadata:
  config_file: "metadata.yaml"
```

```python
rag = RAGWire("config.yaml")
stats = rag.ingest_directory("data/legal/")

# Custom fields are extracted and stored automatically
results = rag.retrieve("data protection policy", filters={"jurisdiction": "EU"})
```

See [Custom Metadata](custom_metadata.md) for the full guide.

---

## Build a Metadata-Aware Filtered Chatbot

Combine auto-filter with a metadata-aware system prompt so the agent knows what data is available and can filter precisely.

```python
from ragwire import RAGWire
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI

rag = RAGWire("config.yaml")

# Discover what's in the collection
values = rag.get_field_values(["company_name", "doc_type", "fiscal_year"])

SYSTEM_PROMPT = f"""
You are a financial document assistant.
Use the search_documents tool to answer questions from the knowledge base.

Available data:
- Companies : {values['company_name']}
- Doc types : {values['doc_type']}
- Fiscal years: {values['fiscal_year']}

The retrieval system automatically filters by company, year, and doc type
when mentioned in the query — you don't need to do anything special.
"""

@tool
def search_documents(query: str) -> str:
    """Search the financial document knowledge base."""
    results = rag.retrieve(query, top_k=5)
    if not results:
        return "No relevant documents found."
    chunks = []
    for doc in results:
        source = doc.metadata.get("file_name", "unknown")
        chunks.append(f"[{source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(chunks)

agent = create_agent(
    model=ChatOpenAI(model="gpt-5.4-nano"),
    tools=[search_documents],
    system_prompt=SYSTEM_PROMPT,
)

response = agent.invoke({
    "messages": [{"role": "user", "content": "What is Apple's net income for 2025?"}]
})
print(response["messages"][-1].content)
```

See [RAG Agent](rag_agent.md) for the full guide including multi-turn memory.
