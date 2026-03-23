# Separate Collections per Client or Domain

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

**When to use this pattern:**

- Multi-tenant SaaS: one collection per customer so data is fully isolated
- Domain isolation: legal, financial, and HR docs are searched independently
- Different embedding models per domain: create each `RAGWire` with a different `config.yaml`

!!! tip "All instances share the same Qdrant server"
    Collections are logically separated inside Qdrant — you don't need a separate database per client.
