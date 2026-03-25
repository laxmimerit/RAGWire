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

| Key | Required | Description |
|---|---|---|
| `name` | Yes | Field key stored in metadata |
| `description` | Yes | Instruction sent to the LLM describing what to extract |
| `type` | No | `string` (default) \| `list` \| `integer` |
| `values` | No | Example/allowed values — shown to LLM as format hints |

!!! note "Open-ended lists"
    For `type: list` fields, `values` are format examples only — not a whitelist. The LLM will extract any value in the same format, even if it's not in the list.

---

## Domain Examples

### Health & Gym Supplement Research Papers

```yaml
fields:
  - name: title
    description: "Full title of the research paper"

  - name: authors
    description: "List of full author names as they appear in the paper"
    type: list

  - name: publication_year
    description: "Year the paper was published"
    type: integer

  - name: supplement_types
    description: "List of all supplements studied in lowercase-hyphenated format. Not limited to the examples — extract any supplement mentioned in the paper. Return empty list if none"
    type: list
    values: ["protein", "creatine", "caffeine", "vitamin-d", "omega-3", "bcaa", "ashwagandha", "magnesium", "beta-alanine"]

  - name: research_focus
    description: "List of all research topics covered in lowercase-hyphenated format. Not limited to the examples — extract any focus area mentioned"
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
  provider: "openai"
  model: "gpt-5.4-nano"

metadata:
  config_file: "metadata.yaml"

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

Explicit filters:

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

Auto-filter (LLM extracts from query):

```python
# LLM extracts {"jurisdiction": "eu", "doc_type": "policy"} automatically
results = rag.retrieve("What are the EU data protection policies?")
```

### 5. Run the example script

A complete runnable script is available at `examples/custom_metadata_usage.py`:

```bash
python examples/custom_metadata_usage.py
```
