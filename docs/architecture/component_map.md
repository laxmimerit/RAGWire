# Component Map

How all modules in the RAGWire package relate to each other — who owns what, who calls whom, and which external libraries each component depends on.

---

## Module Dependency Graph

```mermaid
graph TD
    INIT["ragwire/__init__.py\nPublic API — exports all symbols"]
    INIT --> PIPE

    PIPE["core/pipeline.py\nRAGWire — main orchestrator"]

    PIPE --> CFG["core/config.py\nConfig"]
    PIPE --> LOAD["loaders/markitdown_loader.py\nMarkItDownLoader"]
    PIPE --> SPLIT["processing/splitter.py\nText Splitters"]
    PIPE --> HASH["processing/hashing.py\nSHA256 Hashing"]
    PIPE --> EXT["metadata/extractor.py\nMetadataExtractor"]
    PIPE --> SCH["metadata/schema.py\nDocumentMetadata"]
    PIPE --> EMB["embeddings/factory.py\nget_embedding"]
    PIPE --> QS["vectorstores/qdrant_store.py\nQdrantStore"]
    PIPE --> HYB["retriever/hybrid.py\nget_retriever / hybrid_search"]
    PIPE --> LOG["utils/logging.py\nsetup_logging"]
```

---

## External Library Mapping

| RAGWire Module | Third-Party Libraries | Notes |
|---|---|---|
| `markitdown_loader.py` | `markitdown` | Document → Markdown conversion |
| `splitter.py` | `langchain-text-splitters` | Markdown + recursive splitting |
| `extractor.py` | `langchain-core` (ChatPromptTemplate) | Prompt building + LLM chain |
| `schema.py` | `pydantic` | Metadata schema validation |
| `factory.py` (embeddings) | `langchain-openai` · `langchain-ollama` · `langchain-huggingface` · `langchain-google-genai` | Lazy import — only the configured provider is loaded |
| `qdrant_store.py` | `qdrant-client` · `langchain-qdrant` · `fastembed` | `fastembed` only needed for hybrid search |
| `hybrid.py` | `langchain-qdrant` (QdrantVectorStore) | Similarity / MMR / hybrid retrieval |
| `config.py` | `pyyaml` · `python-dotenv` | YAML loading + env var resolution |
| `pipeline.py` (LLM) | `langchain-openai` · `langchain-ollama` · `langchain-google-genai` · `langchain-groq` · `langchain-anthropic` | Lazy import — only the configured provider is loaded |

---

## RAGWire Class — Internal State

```mermaid
classDiagram
    class RAGWire {
        +config: dict
        +loader: MarkItDownLoader
        +splitter: TextSplitter
        +embedding: EmbeddingModel
        +metadata_extractor: MetadataExtractor
        +vectorstore_wrapper: QdrantStore
        +vectorstore: QdrantVectorStore
        +retriever: Retriever
        -_filter_fields: List[str]
        -_stored_values_cache: dict or None

        +ingest_documents(file_paths) dict
        +ingest_directory(directory) dict
        +retrieve(query, top_k, filters) List[Document]
        +hybrid_search(query, k, filters) List[Document]
        +extract_metadata(text) dict
        +get_field_values(fields, limit) dict
        +filter_fields List[str]
        +discover_metadata_fields() List[str]
        +get_stats() dict

        -_process_document(text, file_path, ...) List[Document]
        -_extract_filters_from_query(query) dict
        -_build_qdrant_filter(filters) Filter
        -_stored_values: dict [property]
        -_initialize_logging()
        -_initialize_loader()
        -_initialize_splitter()
        -_initialize_embeddings()
        -_initialize_llm()
        -_initialize_vectorstore()
        -_initialize_retriever()
    }

    class MetadataExtractor {
        +llm: ChatModel
        +prompt_template: str
        +prompt: ChatPromptTemplate
        +fields: List[str] or None

        +extract(text, stored_values) dict
        +extract_batch(texts, stored_values) List[dict]
        +build_prompt_from_fields(fields)$ str
        +from_yaml(llm, yaml_path)$ MetadataExtractor
        -_parse_json_response(text) dict
    }

    class QdrantStore {
        +client: QdrantClient
        +embedding: EmbeddingModel
        +collection_name: str
        +config: dict

        +set_collection(name)
        +get_store(use_sparse) QdrantVectorStore
        +create_collection(use_sparse)
        +delete_collection()
        +collection_exists() bool
        +file_hash_exists(file_hash) bool
        +get_metadata_keys() List[str]
        +get_field_values(fields, limit) dict
        +create_payload_indexes(fields)
        +get_collection_info() CollectionInfo
    }

    RAGWire --> MetadataExtractor
    RAGWire --> QdrantStore
```

---

## Data Types Flowing Through the Pipeline

```mermaid
flowchart LR
    F["str\nfile path"] -->|"MarkItDownLoader"| MD["str\nmarkdown text"]
    MD -->|"TextSplitter"| CL["List[str]\nchunk texts"]
    CL -->|"MetadataExtractor + metadata dict"| DL["List[Document]\npage_content + metadata"]
    DL -->|"EmbeddingModel + QdrantStore"| VEC["Qdrant points\nvector + payload"]

    Q["str\nquery"] -->|"EmbeddingModel"| QV["List[float]\nquery vector"]
    QV -->|"Retriever"| RES["List[Document]\nranked results"]
```
