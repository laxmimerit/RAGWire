# Ingestion Pipeline

The ingestion pipeline converts raw files into searchable vector chunks stored in Qdrant. It runs once per document (re-ingesting the same file is a no-op thanks to SHA256 deduplication).

---

## Step-by-Step Flow

```mermaid
flowchart TD
    A(["file_paths: List[str]"]) --> B

    B["Compute SHA256\nof file bytes\nsha256_file_from_path()"]
    B --> C{Hash already\nin Qdrant?}

    C -->|Yes| SKIP(["SKIP — already ingested\nskipped += 1"])
    C -->|No| D

    D["Load document\nMarkItDownLoader.load()\nFile → Markdown text"]
    D --> E{Load\nsucceeded?}

    E -->|No| FAIL(["Log error\nfailed += 1"])
    E -->|Yes| F

    F["Split text into chunks\nTextSplitter.split_text()\nchunk_size=10000, overlap=2000"]

    F --> G["Fetch stored collection values\n_stored_values_cache\n(one Qdrant call per run)"]

    G --> H["Extract metadata from chunk[0]\nMetadataExtractor.extract(chunk, stored_values)\nLLM → JSON"]

    H --> I["Attach metadata to every chunk\nsource, file_name, file_type, file_hash\nchunk_id, chunk_hash, chunk_index\ntotal_chunks, created_at\n+ all LLM-extracted fields"]

    I --> J["Generate dense embeddings\nEmbeddingModel.embed_documents(chunks)"]

    J --> K["Upsert into Qdrant\nvectorstore.add_documents(chunks)\nDense vector + sparse vector (if hybrid)"]

    K --> DONE(["processed += 1\nchunks_created += N"])

    DONE --> POST

    POST["After all files:\nget_metadata_keys()\ncreate_payload_indexes()\n_stored_values_cache = None"]
```

---

## Deduplication

```mermaid
flowchart LR
    File["PDF / DOCX / ..."] -->|"read bytes (8KB chunks)"| Hash["SHA256 Hash\n64-char hex string"]
    Hash -->|"scroll filter"| Qdrant[("Qdrant\nmetadata.file_hash")]
    Qdrant -->|"found"| Skip["Skip file"]
    Qdrant -->|"not found"| Ingest["Proceed with ingestion"]
```

The hash is computed by streaming the file in 8 KB chunks — memory-safe for large PDFs. The same hash is stored as `file_hash` in every chunk's metadata, so a single Qdrant scroll query can check for existence.

---

## Chunk Metadata Structure

Every chunk stored in Qdrant carries the following payload:

```
metadata/
├── LLM-extracted (once per document, from chunk[0])
│   ├── company_name      "apple inc."
│   ├── doc_type          "10-k"
│   ├── fiscal_quarter    null
│   └── fiscal_year       [2025]
│
├── File-level (from file system)
│   ├── source            "/data/Apple_10k_2025.pdf"
│   ├── file_name         "Apple_10k_2025.pdf"
│   ├── file_type         "pdf"
│   └── file_hash         "a1b2c3..."
│
└── Chunk-level (per chunk)
    ├── chunk_id          "a1b2c3_0"
    ├── chunk_hash        SHA256(chunk_id + content)
    ├── chunk_index       0
    ├── total_chunks      42
    └── created_at        "2025-01-01T00:00:00+00:00"
```

---

## Text Splitting Strategy

```mermaid
flowchart TD
    Raw["Raw Markdown text\n(from MarkItDown)"]

    Raw --> Strategy{splitter.strategy}

    Strategy -->|markdown| MS["MarkdownTextSplitter\nSplits on: ## → ### → #### → paragraph → sentence"]
    Strategy -->|recursive| RS["RecursiveCharacterTextSplitter\nSplits on: \\n\\n → \\n → space → char"]

    MS --> Chunks["Chunks\n~10,000 chars each\n2,000 char overlap"]
    RS --> Chunks
```

Overlap ensures context is not lost when a sentence spans a chunk boundary.

---

## Post-Ingestion Indexing

After all files are processed, RAGWire creates payload indexes on every metadata field so Qdrant's facet API works for filter extraction:

```mermaid
flowchart LR
    A["get_metadata_keys()\nScroll 1 point → field names"] --> B["create_payload_indexes(fields)"]
    B --> C{Field type}
    C -->|"chunk_index\ntotal_chunks\nfiscal_year"| INT["INTEGER index"]
    C -->|"everything else"| KW["KEYWORD index"]
    INT --> D[("Qdrant payload indexes")]
    KW --> D
    D --> E["_stored_values_cache = None\n(invalidate for next retrieval)"]
```
