# Custom Metadata

RAGWire supports fully custom metadata fields for any domain — legal, HR, medical, e-commerce, and more. Define your fields in a YAML file or pass a custom prompt in code.

---

## Custom Metadata via YAML File

The easiest way to define custom metadata fields is a YAML config file. RAGWire auto-builds the extraction prompt from your field definitions — no code changes needed.

### 1. Create `metadata.yaml`

```yaml
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

### 2. Reference it in `config.yaml`

```yaml
metadata:
  config_file: "metadata.yaml"
```

That's it. RAGWire reads the YAML at startup, builds an extraction prompt from your fields, and uses it for every ingested document. The extracted values are stored in Qdrant and can be filtered at query time exactly like the built-in fields.

### Field definitions

| Key | Required | Description |
|---|---|---|
| `name` | Yes | JSON key name stored in metadata |
| `description` | Yes | Human-readable hint sent to the LLM |
| `values` | No | Allowed values — shown as `val1\|val2\|val3` in the prompt |

### Optional: fully custom prompt

If you need full control over the prompt, add a `prompt_template` key. When present, `fields` is ignored.

```yaml
prompt_template: |
  Extract metadata from the document. Return ONLY valid JSON:
  {
    "organization": "organization name in lowercase",
    "doc_type": "contract|policy|report|memo|other",
    "effective_year": year as integer or null,
    "jurisdiction": "country or region or null"
  }

  Document Text:
  {content}

  Extracted Metadata (JSON only):
```

!!! note "The `{content}` placeholder"
    Your `prompt_template` must include `{content}` — RAGWire substitutes the document text there.

---

## Custom Metadata Extraction in Code

You can also pass a custom prompt directly in Python without a YAML file:

```python
from ragwire import MetadataExtractor
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

Or load directly from a YAML file:

```python
extractor = MetadataExtractor.from_yaml(llm, "metadata.yaml")
metadata = extractor.extract(document_text)
```

!!! note "Custom fields are fully supported"
    Custom fields extracted by your prompt are stored in Qdrant alongside the built-in fields. `DocumentMetadata` accepts arbitrary extra keys, so there are no schema validation errors. You can filter by custom fields using the same `filters={}` mechanism as built-in fields.

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
    values: ["contract", "policy", "report", "memo", "other"]

  - name: effective_year
    description: "Year the document is effective, as integer or null"

  - name: jurisdiction
    description: "Country or region the document applies to, or null"
```

### 2. `config.yaml`

```yaml
embeddings:
  provider: "openai"
  model: "text-embedding-3-small"

llm:
  provider: "openai"
  model: "gpt-5.4-nano"
  temperature: 0.0

metadata:
  config_file: "metadata.yaml"   # ← point to your custom fields

vectorstore:
  url: "http://localhost:6333"
  collection_name: "legal_docs"
  use_sparse: true

retriever:
  search_type: "hybrid"
  top_k: 5
```

### 3. Ingest and retrieve

```python
from ragwire import RAGWire

rag = RAGWire("config.yaml")

# Ingest documents — custom fields extracted from each doc
stats = rag.ingest_directory("data/")
print(f"Ingested {stats['processed']} docs, {stats['chunks_created']} chunks")

# Inspect extracted metadata on a retrieved chunk
results = rag.retrieve("data protection policy", top_k=1)
print(results[0].metadata)
# {
#     "organization": "acme corp",
#     "doc_type": "policy",
#     "effective_year": 2024,
#     "jurisdiction": "EU",
#     "source": "data/acme_data_policy.pdf",
#     "file_name": "acme_data_policy.pdf",
#     ...
# }
```

### 4. Filter by custom fields

Explicit filters:

```python
# Only policy documents
results = rag.retrieve(
    "employee data handling responsibilities",
    filters={"doc_type": "policy"}
)

# EU jurisdiction contracts from 2024
results = rag.retrieve(
    "termination clauses",
    filters={"doc_type": "contract", "jurisdiction": "EU", "effective_year": 2024}
)
```

Auto-filter (LLM extracts from query):

```python
# LLM extracts {"jurisdiction": "EU", "doc_type": "policy"} automatically
results = rag.retrieve("What are the EU data protection policies?")
```

### 5. Run the example script

A complete runnable script is available at `examples/custom_metadata_usage.py`:

```bash
python examples/custom_metadata_usage.py
```
