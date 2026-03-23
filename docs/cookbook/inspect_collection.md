# Inspect What's in Your Collection

Before querying, understand what's actually stored.

## Collection Stats

```python
from ragwire import RAGWire

rag = RAGWire("config.yaml")

stats = rag.get_stats()
print(f"Collection : {stats['collection_name']}")
print(f"Total chunks: {stats['total_documents']}")
print(f"Vector size : {stats['vector_size']}")
```

## Discover Metadata Fields

```python
# What metadata fields exist across all stored documents?
fields = rag.discover_metadata_fields()
print(f"Fields: {fields}")
# → ['company_name', 'doc_type', 'fiscal_year', 'file_name', 'file_type', 'chunk_index', ...]
```

## Get Unique Values per Field

```python
# What values are stored for the key fields?
values = rag.get_field_values(["company_name", "doc_type", "fiscal_year"])
print(values)
# → {
#     'company_name': ['apple', 'microsoft', 'google'],
#     'doc_type':     ['10-k', '10-q'],
#     'fiscal_year':  ['2024', '2025'],
# }
```

`get_field_values` uses Qdrant's native facet API — fast and exact regardless of collection size. Results are ordered by frequency (most common values first).

## Single Field Shorthand

```python
companies = rag.get_field_values("company_name")
# → ['apple', 'microsoft', 'google']
```

## Increase the Value Limit

By default up to 50 unique values are returned per field. Increase for high-cardinality fields:

```python
all_files = rag.get_field_values("file_name", limit=200)
```

## Use Inspection to Build a Smart Agent

Feed the discovered values into your agent's system prompt so it knows what to filter on:

```python
values = rag.get_field_values(["company_name", "doc_type", "fiscal_year"])

system_prompt = f"""
You are a financial document assistant.
Available companies: {values['company_name']}
Available doc types: {values['doc_type']}
Available years: {values['fiscal_year']}
"""
```

See [RAG Agent](../rag_agent.md) for the full metadata-aware agent example.
