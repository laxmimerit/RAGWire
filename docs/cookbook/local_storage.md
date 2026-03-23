# Use Qdrant Without Docker (Local File Storage)

No Docker needed — Qdrant can store vectors in a local folder.

```yaml
vectorstore:
  url: "./qdrant_storage"   # path instead of http://
  collection_name: "my_docs"
  use_sparse: false          # fastembed sparse index not supported in file mode
```

```python
from ragwire import RAGWire

rag = RAGWire("config.yaml")
stats = rag.ingest_directory("data/")
```

The `qdrant_storage/` folder is created automatically. Ideal for local development or single-machine deployments.

!!! note "Hybrid search limitation"
    Local file storage does not support sparse vectors (`use_sparse: false`). Use `search_type: "similarity"` or `"mmr"` instead of `"hybrid"`.
