# Retrieval Pipeline

The retrieval pipeline takes a natural language query and returns the most relevant document chunks from Qdrant. It uses the LLM to automatically extract metadata filters from the query so results are scoped to the right company, year, or document type — without the user having to pass filters manually.

---

## Step-by-Step Flow

```mermaid
flowchart TD
    A(["query: str\ntop_k: int\nfilters: dict or None"])

    A --> B{filters\nprovided?}

    B -->|"Yes (explicit)"| E
    B -->|"No (auto)"| C

    C["Fetch stored values\n_stored_values_cache\n(one Qdrant call if cache empty)"]

    C --> D["LLM extracts filters from query\n_extract_filters_from_query()\nLLM sees actual stored values\ne.g. company_name: ['apple inc.', 'microsoft']"]

    D --> D2{Filters\nextracted?}
    D2 -->|"Yes"| D3["Normalize values to lowercase\ncompany_name: 'Apple' → 'apple inc.'"]
    D2 -->|"No match / error"| E

    D3 --> E

    E["Build Qdrant Filter\n_build_qdrant_filter(filters)\nfield=value → FieldCondition\nlist=[v1,v2] → should (OR logic)"]

    E --> F["Build search_kwargs\nk=top_k\nfilter=qdrant_filter"]

    F --> G["vectorstore.as_retriever()\nsearch_type: similarity / mmr / hybrid"]

    G --> H["Embed query\nEmbeddingModel.embed_query(query)\n→ dense vector"]

    H --> I{search_type}

    I -->|similarity| J["Cosine similarity search\ndense vectors only"]
    I -->|mmr| K["MMR search\ndense vectors\nfetch_k=20 candidates\nre-rank for diversity"]
    I -->|hybrid| L["Hybrid search\ndense + sparse (BM25)\nRetrievalMode.HYBRID\nRRF fusion"]

    J --> M(["List[Document]\npage_content + metadata"])
    K --> M
    L --> M
```

---

## Auto-Filter Extraction Detail

When no filters are passed, the LLM is shown the **exact values stored in Qdrant** and asked to match:

```mermaid
sequenceDiagram
    participant R as retrieve()
    participant C as _stored_values cache
    participant Q as Qdrant facet API
    participant L as LLM

    R->>C: _stored_values (cache miss?)
    alt cache empty
        C->>Q: facet(company_name, doc_type, fiscal_quarter, fiscal_year)
        Q-->>C: {company_name: ['apple inc.', 'microsoft'], ...}
        C-->>R: stored values dict
    else cache hit
        C-->>R: stored values dict (no Qdrant call)
    end

    R->>L: prompt with stored values + user query
    Note over L: "Available values:\n  company_name: ['apple inc.', 'microsoft']\nQuery: What is Apple's Q1 2025 revenue?"
    L-->>R: {"company_name": "apple inc.", "fiscal_quarter": "q1", "fiscal_year": 2025}
    R->>R: normalize to lowercase
    R->>R: build Qdrant filter
```

---

## Filter Building Logic

```mermaid
flowchart TD
    F["filters dict\ne.g. {company_name: 'apple inc.', fiscal_year: [2023, 2024]}"]

    F --> Loop["For each key: value"]

    Loop --> TypeCheck{value type}

    TypeCheck -->|"str or int\ne.g. company_name: 'apple inc.'"| Single["FieldCondition\nkey=metadata.company_name\nmatch=MatchValue('apple inc.')"]

    TypeCheck -->|"list\ne.g. fiscal_year: [2023, 2024]"| Multi["Filter(should=[\n  FieldCondition(match=2023),\n  FieldCondition(match=2024)\n])\n→ OR logic within field"]

    Single --> Must["Filter(must=[...])\n→ AND logic across fields"]
    Multi --> Must

    Must --> Qdrant[("Qdrant filtered search\ncompany='apple inc.'\nAND (year=2023 OR year=2024)")]
```

---

## Hybrid Search Internals

```mermaid
flowchart LR
    Query["Query text"]

    Query -->|"embed_query()"| Dense["Dense vector\n(semantic meaning)\nOllama / OpenAI / etc."]
    Query -->|"FastEmbedSparse()"| Sparse["Sparse vector\n(BM25 keyword weights)\nFastEmbed"]

    Dense --> Qdrant[("Qdrant\nRetrievalMode.HYBRID")]
    Sparse --> Qdrant

    Qdrant -->|"Reciprocal Rank Fusion"| RRF["RRF score\n= dense_rank + sparse_rank"]
    RRF --> Results["Top-K ranked chunks"]
```

Hybrid search is only active when `use_sparse: true` in config and `fastembed` is installed. If `fastembed` is missing, RAGWire falls back to dense-only search with a warning.

---

## Stored Values Cache Lifecycle

```mermaid
stateDiagram-v2
    [*] --> Empty : RAGWire() init

    Empty --> Populated : First retrieve() or extract_metadata() call\n→ Qdrant facet API called

    Populated --> Populated : Subsequent retrieve() calls\n(no Qdrant call — cache hit)

    Populated --> Empty : ingest_documents() completes\n_stored_values_cache = None

    Empty --> Populated : Next retrieve() call\n→ Qdrant facet API called again\n(now includes newly ingested data)
```
