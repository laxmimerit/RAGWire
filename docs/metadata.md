# Metadata — Schema, Extraction & Filtering

Understanding metadata is critical for building precise RAG applications. RAGWire automatically extracts and stores metadata on every chunk — and you can filter by it at query time.

---

## Metadata Schema

Every chunk stored in Qdrant carries the following metadata fields:

### LLM-Extracted Fields
These are extracted once per document from the first chunk using your configured LLM.

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
| `created_at` | `str` | `"2026-03-22T10:00:00"` | ISO timestamp of ingestion |

---

## Inspecting Metadata on Retrieved Chunks

```python
from ragwire import RAGPipeline

pipeline = RAGPipeline("config.yaml")
results = pipeline.retrieve("What is Apple's revenue?", top_k=3)

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

## Filtering at Query Time

Pass a `filters` dictionary to `retrieve()` to narrow results to specific metadata values.

### Filter by company

```python
results = pipeline.retrieve(
    "What is the total revenue?",
    top_k=5,
    filters={"company_name": "apple"}
)
```

### Filter by document type

```python
results = pipeline.retrieve(
    "What are the risk factors?",
    top_k=5,
    filters={"doc_type": "10-k"}
)
```

### Filter by fiscal year

```python
# fiscal_year is stored as a list — pass the year as an int
results = pipeline.retrieve(
    "What is the net income?",
    top_k=5,
    filters={"fiscal_year": 2025}
)
```

### Filter by company + year (combined)

```python
results = pipeline.retrieve(
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
results = pipeline.retrieve(
    "What are the capital expenditures?",
    top_k=5,
    filters={"file_name": "Apple_10k_2025.pdf"}
)
```

---

## Filters in Hybrid Search

Filters also work with `hybrid_search()`:

```python
results = pipeline.hybrid_search(
    "Apple revenue fiscal 2025",
    k=5,
    filters={"company_name": "apple"}
)
```

---

## Telling an External LLM What Metadata Is Available

When building a RAG chatbot, pass the schema to your LLM so it can decide what filters to apply before querying.

```python
METADATA_SCHEMA = """
Available metadata fields for filtering RAG queries:

LLM-extracted (from document content):
- company_name (str): Company name in lowercase. Example: "apple", "microsoft"
- doc_type (str): Document type. Values: "10-k", "10-q", "8-k"
- fiscal_quarter (str or null): Quarter. Values: "q1", "q2", "q3", "q4"
- fiscal_year (list[int]): Fiscal year(s). Example: [2025]

File-level:
- file_name (str): Original filename. Example: "Apple_10k_2025.pdf"
- file_type (str): File extension. Example: "pdf"
- chunk_index (int): Chunk position within the document (0-based)
- total_chunks (int): Total chunks in the document

Use these fields to filter retrieved chunks when the user query specifies
a particular company, document type, or time period.
"""

SYSTEM_PROMPT = f"""
You are a financial document assistant.
You have access to a RAG pipeline with the following metadata schema:

{METADATA_SCHEMA}

When answering questions, extract any filters from the user query
(e.g. company name, year, document type) and apply them.
"""
```

### Full example — LLM-driven filter extraction

```python
from ragwire import RAGPipeline
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import json

pipeline = RAGPipeline("config.yaml")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

FILTER_PROMPT = """
Given the user query below, extract metadata filters as JSON.
Only include fields if clearly mentioned. Return {} if no filters apply.

Available fields: company_name (str), doc_type (str), fiscal_year (int), fiscal_quarter (str)

User query: {query}

Filters (JSON only):
"""

def query_with_filters(user_query: str, top_k: int = 5):
    # Step 1: Ask LLM to extract filters from the query
    prompt = ChatPromptTemplate.from_template(FILTER_PROMPT)
    chain = prompt | llm
    response = chain.invoke({"query": user_query})

    try:
        filters = json.loads(response.content.strip())
    except Exception:
        filters = {}

    print(f"Extracted filters: {filters}")

    # Step 2: Retrieve with filters
    results = pipeline.retrieve(user_query, top_k=top_k, filters=filters or None)

    return results


# Example usage
results = query_with_filters("What is Apple's revenue for fiscal year 2025?")
for doc in results:
    print(doc.metadata.get("company_name"), doc.metadata.get("fiscal_year"))
    print(doc.page_content[:200])
    print()
```

---

## Custom Metadata Extraction Prompt

If your documents are not financial (e.g. legal, medical, technical), you can customize the extraction prompt:

```python
from ragwire.metadata.extractor import MetadataExtractor
from langchain_openai import ChatOpenAI

custom_prompt = """
Extract metadata from the following document. Return ONLY valid JSON:
{{
  "organization": "organization name in lowercase",
  "doc_type": "contract|policy|report|other",
  "effective_year": year as integer or null,
  "jurisdiction": "country or region or null"
}}

Document Text:
{content}

Extracted Metadata (JSON only):
"""

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
extractor = MetadataExtractor(llm, prompt_template=custom_prompt)

metadata = extractor.extract(document_text)
print(metadata)
```

!!! warning "Custom fields and filtering"
    If you add custom fields via a custom prompt, they will be stored in Qdrant but are not part of the default `DocumentMetadata` schema. You can still filter by them using the same `filters={}` mechanism.
