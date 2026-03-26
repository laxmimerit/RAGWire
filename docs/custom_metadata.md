# Custom Metadata

RAGWire supports fully custom metadata fields for any domain — legal, HR, medical, e-commerce, and more. Define your fields in a YAML file and RAGWire builds a typed Pydantic schema automatically.

---

## Custom Metadata via YAML File

The easiest way to define custom metadata fields is a YAML config file. RAGWire builds a Pydantic model from your field definitions and uses `with_structured_output` for reliable, type-safe extraction — no manual JSON parsing.

### 1. Create `metadata.yaml`

```yaml
fields:
  - name: organization
    description: "Organization name in lowercase"

  - name: doc_type
    description: "Type of document"
    values: ["contract", "policy", "report", "memo"]

  - name: effective_year
    description: "Year the document is effective"
    type: integer

  - name: jurisdiction
    description: "Country or region the document applies to, or null"
```

### 2. Reference it in `config.yaml`

```yaml
metadata:
  config_file: "metadata.yaml"
```

That's it. RAGWire reads the YAML at startup, builds a Pydantic schema from your fields, and uses it for every ingested document. The extracted values are stored in Qdrant and can be filtered at query time exactly like the built-in fields.

See [Domain Examples](#domain-examples) below for ready-to-use schemas for legal, HR, healthcare, and more.

### Field definitions

**Top-level keys**

| Key | Required | Description |
|---|---|---|
| `prompt` | No | Custom extraction prompt. Document text is injected automatically — no placeholder needed. Defaults to the built-in RAGWire prompt if omitted. |
| `fields` | Yes | List of metadata field definitions (see below) |


**Field definition keys**

| Key | Required | Description |
|---|---|---|
| `name` | Yes | Field key stored in metadata |
| `description` | Yes | Instruction sent to the LLM describing what to extract |
| `type` | No | `string` (default) \| `list` \| `integer` |
| `values` | No | Example/allowed values — shown to LLM as format hints |

!!! note "Open-ended lists"
    For `type: list` fields, `values` are format examples only — not a whitelist. The LLM will extract any value in the same format, even if it's not in the list.

---

## Custom Extraction Prompt

By default RAGWire uses a built-in extraction prompt. You can override it per-config using a top-level `prompt` key in your `metadata.yaml`. Write your instructions — the document text is appended automatically.

```yaml
prompt: |
  You are a <domain> expert. Extract the metadata fields below from the document.
  Be thorough — only return null if the information is genuinely absent.

fields:
  - name: my_field
    description: "What to extract"
```

!!! note
    The `prompt` key is optional. When omitted, the default RAGWire extraction prompt is used. No `{content}` placeholder needed — document text is appended automatically.

---

## Domain Examples

### Health & Gym Supplement Research Papers

```yaml
prompt: |
  You are an expert research paper analyst specializing in health, fitness, and sports science.
  Your task is to extract structured metadata from the research paper below.

  ## Extraction Rules
  1. **Be thorough**: Extract every field you can find. A field should only be null if the information is completely absent.
  2. **Be precise**: Extract exactly what is stated. Do not infer, assume, or hallucinate values not present in the document.
  3. **Lists**: Scan the entire document and extract ALL matching values — not just the first occurrence.
  4. **Strings**: Normalize to lowercase. Trim extra whitespace.
  5. **Integers**: Return the numeric value only — no units, symbols, or surrounding text.
  6. **Null**: Return null only when the field is genuinely not mentioned anywhere in the document.

fields:
  - name: title
    description: "Full title of the research paper exactly as it appears in the document. Do not paraphrase."

  - name: authors
    description: "List of full author names exactly as they appear in the paper (e.g. 'John A. Smith'). Extract all authors."
    type: list

  - name: publication_year
    description: "Year the paper was published or last revised. Extract the 4-digit year only."
    type: integer

  - name: research_focus
    description: "List of all primary research topics covered, in lowercase-hyphenated format. Not limited to the examples — extract any focus area mentioned in the paper."
    type: list
    values: ["muscle-growth", "recovery", "performance", "endurance", "cognitive-function", "fat-loss", "safety", "hormonal"]
```

---

### Legal / HR

```yaml
fields:
  - name: organization
    description: "Organization name in lowercase"

  - name: doc_type
    description: "Type of document"
    values: ["contract", "policy", "report", "memo", "nda", "agreement"]

  - name: effective_year
    description: "Year the document is effective"
    type: integer

  - name: jurisdiction
    description: "Country or region the document applies to, or null"

  - name: parties
    description: "List of parties involved in the document (individuals or organizations)"
    type: list

  - name: status
    description: "Current status of the document"
    values: ["active", "expired", "draft", "terminated"]
```

### Healthcare / Medical

```yaml
fields:
  - name: condition
    description: "List of medical conditions or diseases discussed, in lowercase-hyphenated format"
    type: list

  - name: treatment_type
    description: "List of treatments or interventions discussed"
    type: list
    values: ["medication", "surgery", "therapy", "diagnostic", "preventive", "rehabilitation"]

  - name: specialty
    description: "List of medical specialties the document covers"
    type: list
    values: ["cardiology", "oncology", "neurology", "orthopedics", "general-practice", "psychiatry"]

  - name: publication_year
    description: "Year the document was published"
    type: integer

  - name: patient_population
    description: "Target patient population described in the document, or null"

  - name: evidence_level
    description: "Level of clinical evidence"
    values: ["systematic-review", "rct", "cohort-study", "case-study", "expert-opinion"]

  - name: key_finding
    description: "Most important clinical finding or recommendation, maximum 200 characters"
```

---

## Complete Custom Metadata Example

Here is a full end-to-end workflow for a **legal/HR** use case using custom metadata fields.

### 1. `metadata.yaml`

```yaml
fields:
  - name: organization
    description: "Organization name in lowercase"

  - name: doc_type
    description: "Type of document"
    values: ["contract", "policy", "report", "memo"]

  - name: effective_year
    description: "Year the document is effective"
    type: integer

  - name: jurisdiction
    description: "Country or region the document applies to, or null"

  - name: parties
    description: "List of parties involved in the document (individuals or organizations)"
    type: list
```

### 2. `config.yaml`

```yaml
embeddings:
  provider: "openai"
  model: "text-embedding-3-small"

llm:
  provider: "ollama"
  model: "qwen3.5:9b"
  base_url: "http://localhost:11434"

metadata:
  config_file: "metadata.yaml"

vectorstore:
  url: "http://localhost:6333"
  collection_name: "legal_docs"
  use_sparse: true

retriever:
  search_type: "hybrid"
  top_k: 5
  auto_filter: false   # keep false for agent use — use get_filter_context() instead
```

### 3. Ingest and retrieve

```python
from ragwire import RAGWire

rag = RAGWire("config.yaml")

stats = rag.ingest_directory("data/")
print(f"Ingested {stats['processed']} docs, {stats['chunks_created']} chunks")

results = rag.retrieve("data protection policy", top_k=1)
print(results[0].metadata)
# {
#     "organization": "acme corp",
#     "doc_type": "policy",
#     "effective_year": 2024,
#     "jurisdiction": "eu",
#     "source": "data/acme_data_policy.pdf",
#     ...
# }
```

### 4. Filter by custom fields

Explicit filters — pass a dict directly, no LLM extraction:

```python
results = rag.retrieve(
    "employee data handling responsibilities",
    filters={"doc_type": "policy"}
)

results = rag.retrieve(
    "termination clauses",
    filters={"doc_type": "contract", "jurisdiction": "eu", "effective_year": 2024}
)
```

Agent-controlled — expose `get_filter_context` and `search_documents` as two separate tools. The agent calls `get_filter_context` first to understand available metadata, then decides what filters to pass to `search_documents`:

```python
from typing import Optional

@tool
def get_filter_context(query: str) -> str:
    """Get available metadata fields, stored values, and filter suggestions for a query.
    Call this before search_documents when the query involves specific metadata.
    """
    return rag.get_filter_context(query)

@tool
def search_documents(query: str, filters: Optional[dict] = None) -> str:
    """Search the document knowledge base with optional filters."""
    results = rag.retrieve(query, top_k=5, filters=filters)
    if not results:
        return "No relevant documents found."
    chunks = []
    for doc in results:
        source = doc.metadata.get("file_name", "unknown")
        meta_parts = [
            f"{k}={str(v)[:100]}"
            for k, v in doc.metadata.items()
            if k != "file_name" and v not in (None, "", [])
        ]
        header = f"[{source}" + (f" | {', '.join(meta_parts)}" if meta_parts else "") + "]"
        chunks.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(chunks)

# Agent flow:
# 1. get_filter_context("EU data protection policies")
#    → sees jurisdiction: ["eu", "us"], doc_type: ["policy", "contract"]
#    → extracted: {"jurisdiction": "eu", "doc_type": "policy"}
# 2. search_documents("EU data protection policies", filters={"jurisdiction": "eu", "doc_type": "policy"})
```

### 5. Run the example script

A complete runnable script is available at `examples/custom_metadata_usage.py`:

```bash
python examples/custom_metadata_usage.py
```
