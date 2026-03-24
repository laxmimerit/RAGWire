# Component Map

How all modules in the RAGWire package relate to each other — who owns what, who calls whom, and which external libraries each component depends on.

---

## Module Dependency Graph

```mermaid
graph TB
    INIT["ragwire/__init__.py\nPublic exports"]

    subgraph Core ["Core"]
        direction LR
        PIPE["pipeline.py\nRAGWire orchestrator"]
        CFG["config.py\nYAML + env vars"]
    end

    subgraph DocProc ["Document Processing"]
        direction LR
        LOAD["markitdown_loader.py"] --- SPLIT["splitter.py"] --- HASH["hashing.py"]
    end

    subgraph Intel ["Intelligence"]
        direction LR
        EXT["extractor.py\nMetadataExtractor"] --- SCH["schema.py\nDocumentMetadata"]
    end

    subgraph Store ["Storage"]
        direction LR
        EMB["factory.py\nget_embedding"] --- QS["qdrant_store.py\nQdrantStore"]
    end

    subgraph RetUtil ["Retrieval & Utilities"]
        direction LR
        HYB["hybrid.py"] --- LOG["logging.py"]
    end

    INIT --> Core
    INIT --> DocProc
    INIT --> Intel
    INIT --> Store
    INIT --> RetUtil

    PIPE --> CFG
    PIPE --> DocProc
    PIPE --> Intel
    PIPE --> Store
    PIPE --> HYB
    PIPE --> LOG
```

---

## External Library Mapping

```mermaid
graph TB
    subgraph RAGWire ["RAGWire Modules"]
        direction LR
        LOAD["MarkItDownLoader"] --- SPLIT["Text Splitters"]
        EXT["MetadataExtractor"] --- EMB["get_embedding"]
        QS["QdrantStore"] --- HYB["hybrid_search"]
        CFG["Config"] --- LOG["logging.py"]
    end

    subgraph CoreLibs ["Core Libraries"]
        direction LR
        MID["markitdown"] --- LTS["langchain-text-splitters"]
        LCC["langchain-core"] --- QC["qdrant-client"]
        LQ["langchain-qdrant"] --- FE["fastembed"]
        YML["pyyaml"] --- DOT["python-dotenv"]
    end

    subgraph EmbedProviders ["Embedding Providers (lazy)"]
        direction LR
        OAI["langchain-openai"] --- HF["langchain-huggingface"]
        OLL["langchain-ollama"] --- GGL["langchain-google-genai"]
    end

    subgraph LLMProviders ["LLM Providers (lazy)"]
        direction LR
        OAI2["langchain-openai"] --- OLL2["langchain-ollama"]
        GGL2["langchain-google-genai"] --- GRQ["langchain-groq"] --- ANT["langchain-anthropic"]
    end

    LOAD --> MID
    SPLIT --> LTS
    EXT --> LCC
    CFG --> YML
    CFG --> DOT
    QS --> QC
    QS --> LQ
    QS --> FE
    HYB --> LQ
    EMB -.-> EmbedProviders
    EXT -.-> LLMProviders
```

Dashed lines = lazy imports (only loaded when that provider is configured).

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
