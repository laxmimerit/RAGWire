# Metadata — Schema, Extraction & Filtering

Understanding metadata is critical for building precise RAG applications. RAGWire automatically extracts and stores metadata on every chunk — and you can filter by it at query time.

---

## Metadata Schema

Every chunk stored in Qdrant carries the following metadata fields:

### LLM-Extracted Fields
These are extracted once per document from the first chunk using your configured LLM.

!!! note "Default schema — Finance"
    The fields below (`company_name`, `doc_type`, `fiscal_quarter`, `fiscal_year`) are the **default** metadata schema, designed for financial documents. You are not locked into these fields.
    RAGWire lets you define any fields you need — product names, departments, case IDs, languages, or anything else — via a simple YAML file. See [Custom Metadata](custom_metadata.md) for details.

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
| `rag.filter_fields` | Semantic/LLM-extracted fields only | Building filter prompts, agent system prompts |
| `rag.discover_metadata_fields()` | All fields including system fields | Collection inspection, debugging |

```python
# Filterable fields only — use these for filter prompts
rag.filter_fields
# → ['company_name', 'doc_type', 'fiscal_quarter', 'fiscal_year']

# All fields — includes file_hash, chunk_id, source, created_at, etc.
rag.discover_metadata_fields()
# → ['company_name', 'doc_type', 'fiscal_year', 'file_name', 'file_hash', 'chunk_id', ...]

# Get values for filterable fields
rag.get_field_values(rag.filter_fields)
# → {'company_name': ['apple', 'microsoft'], 'doc_type': ['10-k', '10-q'], ...}

# Raise the limit for high-cardinality fields
rag.get_field_values("file_name", limit=200)
# → ['Apple_10k_2025.pdf', 'Microsoft_10k_2025.pdf', ...]
```

Results are ordered by frequency — most common values first. The default `limit=50` covers most use cases. Increase it for high-cardinality fields like `file_name`.

This is especially useful when building an LLM agent — pass the filterable fields and values into the system prompt so the agent knows exactly what to filter by:

```python
fields = rag.discover_metadata_fields()
values = rag.get_field_values(["company_name", "doc_type", "fiscal_year"])

SYSTEM_PROMPT = f"""
You have access to a financial document RAG pipeline.

Available metadata fields and values:
- company_name: {values['company_name']}
- doc_type: {values['doc_type']}
- fiscal_year: {values['fiscal_year']}

Use these to apply precise filters when answering questions.
"""
```

---

## Filtering at Query Time

RAGWire applies filters in two ways:

- **Explicit filters** — pass a `filters` dict directly; LLM extraction is skipped
- **Auto-filter** — pass nothing; the configured LLM extracts filters from the query automatically

```python
# Explicit — LLM extraction skipped
results = rag.retrieve("What is the revenue?", filters={"company_name": "apple"})

# Auto — LLM extracts {"company_name": "apple", "fiscal_year": 2025} from the query
results = rag.retrieve("What is Apple's revenue for 2025?")
```

The available filter fields match your metadata schema. By default these are `company_name`, `doc_type`, `fiscal_quarter`, and `fiscal_year`. If you've configured [custom metadata](custom_metadata.md), auto-filter will use your custom fields instead.

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

# Multiple years — pass as list; matches documents covering ANY of the years (OR logic)
results = rag.retrieve(
    "Compare net income across 2023 and 2024",
    top_k=10,
    filters={"fiscal_year": [2023, 2024]}
)
```

### Filter by company + year (combined)

```python
results = rag.retrieve(
    "What is the revenue breakdown by segment?",
    top_k=5,
    filters={
        "company_name": "apple",
        "fiscal_year": 2025
    }
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

Filters also work with `hybrid_search()`:

```python
results = rag.hybrid_search(
    "Apple revenue fiscal 2025",
    k=5,
    filters={"company_name": "apple"}
)
```

---

## Telling an External LLM What Metadata Is Available

When building a RAG chatbot, pass the schema to your LLM so it can decide what filters to apply before querying.

Instead of hardcoding field names and values in your system prompt, build it dynamically from the collection. This way the prompt is always accurate and works for any metadata schema:

```python
from ragwire import RAGWire

rag = RAGWire("config.yaml")

# filter_fields returns only semantic/filterable fields (e.g. company_name, doc_type)
# Use this instead of discover_metadata_fields() which includes system fields
# like file_hash, chunk_id, source that are not useful for filtering
fields = rag.filter_fields
values = rag.get_field_values(fields)

field_descriptions = "\n".join(
    f"- {field}: {values[field]}" if values.get(field) else f"- {field}"
    for field in fields
)

SYSTEM_PROMPT = f"""
You are a document assistant with access to a RAG pipeline.

Available metadata fields and known values:
{field_descriptions}

When answering questions, extract any filters from the user query and apply them.
"""
```

This generates a prompt grounded in real data — no hardcoded assumptions about field names, types, or allowed values.

### Full example — LLM-driven filter extraction

The prompt is built **dynamically** from what is actually in your collection — no hardcoded field names, types, or values. This works identically whether you use the default finance schema or [custom metadata](custom_metadata.md).

```python
from ragwire import RAGWire
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import json

rag = RAGWire("config.yaml")
llm = ChatOpenAI(model="gpt-5.4-nano")

# filter_fields returns only the semantic fields used for filtering
# (excludes system fields like file_hash, chunk_id, source, created_at)
fields = rag.filter_fields
values = rag.get_field_values(fields)

# Build the filter prompt dynamically — works for any schema
field_descriptions = "\n".join(
    f"- {field}: {values[field]}" if values.get(field) else f"- {field}"
    for field in fields
)

FILTER_PROMPT = f"""
Given the user query below, extract metadata filters as JSON.
Only include fields if clearly mentioned in the query. Return {{}} if no filters apply.

Available metadata fields and known values:
{field_descriptions}

User query: {{query}}

Filters (JSON only):
"""

def query_with_filters(user_query: str, top_k: int = 5):
    # Step 1: LLM extracts filters from the query using the dynamic prompt
    prompt = ChatPromptTemplate.from_template(FILTER_PROMPT)
    chain = prompt | llm
    response = chain.invoke({"query": user_query})

    try:
        filters = json.loads(response.content.strip())
    except Exception:
        filters = {}

    print(f"Extracted filters: {filters}")

    # Step 2: Retrieve with extracted filters
    results = rag.retrieve(user_query, top_k=top_k, filters=filters or None)
    return results


# Works for any domain — finance, legal, medical, HR, etc.
results = query_with_filters("What is Apple's revenue for fiscal year 2025?")
for doc in results:
    print(doc.metadata)
    print(doc.page_content[:200])
    print()
```

The prompt RAGWire sends to the LLM will look like this at runtime (example with default finance schema):

```
Available metadata fields and known values:
- company_name: ['apple', 'microsoft', 'google']
- doc_type: ['10-k', '10-q']
- fiscal_year: [2023, 2024, 2025]
- fiscal_quarter: ['q1', 'q2', 'q3', 'q4']
- file_name: ['Apple_10k_2025.pdf', 'Microsoft_10k_2025.pdf']
...
```

The LLM sees the actual values in your collection, so it can match them precisely instead of guessing.

---

For custom metadata fields (legal, HR, medical, or any non-financial domain), see [Custom Metadata](custom_metadata.md).
