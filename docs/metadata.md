# Metadata — Schema, Extraction & Filtering

Understanding metadata is critical for building precise RAG applications. RAGWire automatically extracts and stores metadata on every chunk — and you can filter by it at query time.

---

## Metadata Schema

Every chunk stored in Qdrant carries the following metadata fields:

### LLM-Extracted Fields
Extracted once per document from the first chunk using your configured LLM.

!!! note "Default schema — Finance"
    The fields below are the **default** metadata schema, designed for financial documents. You are not locked into these fields.
    RAGWire lets you define any fields you need via a simple YAML file. See [Custom Metadata](custom_metadata.md) for details.

| Field | Type | Example | Description |
|---|---|---|---|
| `company_name` | `str` | `"apple"` | Company name, normalized to lowercase |
| `doc_type` | `str` | `"10-k"` | Document type (`10-k`, `10-q`, `8-k`) |
| `fiscal_quarter` | `str` | `"q1"` | Fiscal quarter (`q1`–`q4`) or `null` |
| `fiscal_year` | `list[int]` | `[2025]` | Fiscal year(s) covered by the document |

### File-Level Fields
Set automatically from the file at ingestion time.

| Field | Type | Example | Description |
|---|---|---|---|
| `source` | `str` | `"/data/Apple_10k_2025.pdf"` | Full file path |
| `file_name` | `str` | `"Apple_10k_2025.pdf"` | Original filename |
| `file_type` | `str` | `"pdf"` | File extension |
| `file_hash` | `str` | `"abc123..."` | SHA256 hash — used for deduplication |

### Chunk-Level Fields
Set per chunk at ingestion time.

| Field | Type | Example | Description |
|---|---|---|---|
| `chunk_id` | `str` | `"abc123_0"` | Unique chunk identifier (`file_hash_index`) |
| `chunk_hash` | `str` | `"def456..."` | SHA256 hash of chunk content |
| `chunk_index` | `int` | `0` | Position of this chunk within the document |
| `total_chunks` | `int` | `42` | Total chunks in the document |
| `created_at` | `str` | `"2026-03-22T10:00:00+00:00"` | UTC ISO timestamp set at ingestion |

---

## Inspecting Metadata on Retrieved Chunks

```python
from ragwire import RAGWire

rag = RAGWire("config.yaml")
results = rag.retrieve("What is Apple's revenue?", top_k=3)

for doc in results:
    print(doc.metadata)
```

Example output:

```python
{
    "company_name": "apple",
    "doc_type": "10-k",
    "fiscal_quarter": None,
    "fiscal_year": [2025],
    "source": "/data/Apple_10k_2025.pdf",
    "file_name": "Apple_10k_2025.pdf",
    "file_type": "pdf",
    "file_hash": "a1b2c3...",
    "chunk_id": "a1b2c3_0",
    "chunk_hash": "d4e5f6...",
    "chunk_index": 0,
    "total_chunks": 42,
    "created_at": "2026-03-22T10:00:00.000000"
}
```

---

## Discovering Available Fields and Values

RAGWire provides two ways to inspect fields — use the right one for your purpose:

| Method | Returns | Use for |
|---|---|---|
| `rag.filter_fields` | Semantic/LLM-extracted fields only | Building filter prompts, agent prompts |
| `rag.discover_metadata_fields()` | All fields including system fields | Collection inspection, debugging |

```python
# Filterable fields only — use these for filter prompts
rag.filter_fields
# → ['company_name', 'doc_type', 'fiscal_quarter', 'fiscal_year']

# All fields — includes file_hash, chunk_id, source, created_at, etc.
rag.discover_metadata_fields()
# → ['company_name', 'doc_type', 'fiscal_year', 'file_name', 'file_hash', 'chunk_id', ...]

# Get stored values for filterable fields
rag.get_field_values(rag.filter_fields)
# → {'company_name': ['apple', 'microsoft'], 'doc_type': ['10-k', '10-q'], ...}

# Raise the limit for high-cardinality fields (default limit=50)
rag.get_field_values("file_name", limit=200)
# → ['Apple_10k_2025.pdf', 'Microsoft_10k_2025.pdf', ...]
```

Results are ordered by frequency — most common values first.

---

## Filtering at Query Time

RAGWire supports three filtering modes:

| Mode | How | When to use |
|---|---|---|
| **Explicit** | Pass `filters=` dict to `retrieve()` | Programmatic pipelines, known inputs |
| **Auto-filter** | Set `auto_filter: true` in config | Simple chatbots, no agent involved |
| **Agent-controlled** | Call `extract_filters()` or `get_filter_context()` manually | Agents that need to reason about filters |

!!! note "auto_filter is off by default"
    No filter extraction happens automatically unless `auto_filter: true` is set in `config.yaml`. For agents, keep the default and use `extract_filters()` or `get_filter_context()` to control extraction explicitly.

```python
# Explicit — LLM extraction skipped entirely
results = rag.retrieve("What is the revenue?", filters={"company_name": "apple"})

# Auto-filter — requires auto_filter: true in config.yaml
results = rag.retrieve("What is Apple's revenue for 2025?")

# Agent-controlled — extract, inspect, adjust, then retrieve
filters = rag.extract_filters("What is Apple's revenue for 2025?")
# → {"company_name": "apple", "fiscal_year": 2025}
results = rag.retrieve("What is Apple's revenue for 2025?", filters=filters)
```

---

### Filter by company

```python
results = rag.retrieve(
    "What is the total revenue?",
    top_k=5,
    filters={"company_name": "apple"}
)
```

### Filter by document type

```python
results = rag.retrieve(
    "What are the risk factors?",
    top_k=5,
    filters={"doc_type": "10-k"}
)
```

### Filter by fiscal year

```python
# Single year — pass as int
results = rag.retrieve(
    "What is the net income?",
    top_k=5,
    filters={"fiscal_year": 2025}
)

# Multiple years — matches documents covering ANY of the years (OR logic)
results = rag.retrieve(
    "Compare net income across 2023 and 2024",
    top_k=10,
    filters={"fiscal_year": [2023, 2024]}
)
```

### Combined filters

```python
results = rag.retrieve(
    "What is the revenue breakdown by segment?",
    top_k=5,
    filters={"company_name": "apple", "fiscal_year": 2025}
)
```

### Filter by file name

```python
results = rag.retrieve(
    "What are the capital expenditures?",
    top_k=5,
    filters={"file_name": "Apple_10k_2025.pdf"}
)
```

---

## Filters in Hybrid Search

Filters work identically with `hybrid_search()`:

```python
results = rag.hybrid_search(
    "Apple revenue fiscal 2025",
    k=5,
    filters={"company_name": "apple"}
)
```

---

## Agent-Controlled Filtering

For agents, keep `auto_filter` off and use two tools — one for metadata awareness, one for retrieval:

### Two-tool pattern (recommended)

```python
from typing import Optional

@tool
def get_filter_context(query: str) -> str:
    """Get available metadata fields, stored values, and filter suggestions for a query.

    Call this before search_documents when the query involves specific metadata
    (company, year, document type, etc.). Use it to decide what filters to apply.
    Safe to call per sub-query in multi-query flows — always fresh from Qdrant.
    """
    return rag.get_filter_context(query)

@tool
def search_documents(query: str, filters: Optional[dict] = None) -> str:
    """Search the document knowledge base. Pass filters decided from get_filter_context."""
    results = rag.retrieve(query, top_k=5, filters=filters)
    if not results:
        return "No relevant documents found."
    return "\n\n---\n\n".join(
        f"[{doc.metadata.get('file_name', 'unknown')}]\n{doc.page_content}"
        for doc in results
    )
```

Agent flow:
```
1. Agent calls get_filter_context("Apple revenue 2025")
   → sees fields, stored values, extracted: {"company_name": "apple", "fiscal_year": 2025}
   → decides filters

2. Agent calls search_documents("Apple revenue 2025", filters={"company_name": "apple", "fiscal_year": 2025})
```

The agent calls `get_filter_context` only when metadata is relevant — skips it for purely semantic queries. For multi-query tasks each sub-query gets its own fresh context.

### What `get_filter_context()` returns

```
## RAGWire Filter Context

### Available Metadata Fields and Stored Values
- **company_name**: ['apple', 'microsoft', 'google']
- **doc_type**: ['10-k', '10-q']
- **fiscal_year**: [2023, 2024, 2025]

### Extracted Filters from Query
- **company_name**: `apple`
- **fiscal_year**: `2025`

### Instructions
1. Review the extracted filters above.
2. If an extracted value does not match or closely relate to any stored value, adjust or drop that filter.
3. If the query has no clear metadata intent, pass an empty dict {} as filters.
4. Pass the final filters dict to the retrieval tool as filters=.
```

---

For custom metadata fields (legal, HR, medical, or any non-financial domain), see [Custom Metadata](custom_metadata.md).
