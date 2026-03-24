# Metadata Extraction

Metadata extraction is the intelligence layer of RAGWire. It uses an LLM to read the first chunk of every document and produce structured JSON fields that are later used for precise filtering at retrieval time.

---

## Where Metadata Extraction Fits

```mermaid
flowchart LR
    subgraph Ingestion
        Chunk0["chunk[0]\n(first ~10,000 chars)"]
        Extractor["MetadataExtractor\n.extract(text, stored_values)"]
        Meta["metadata dict\n{company_name, doc_type, ...}"]
        AllChunks["All chunks\n(same metadata attached to each)"]
        Chunk0 --> Extractor --> Meta --> AllChunks
    end

    subgraph Retrieval
        Query["User query"]
        FilterExtractor["_extract_filters_from_query()\n(same LLM, different prompt)"]
        Filters["filters dict\n{company_name: 'apple inc.', ...}"]
        Query --> FilterExtractor --> Filters
    end
```

---

## Extraction Flow (Ingestion Time)

```mermaid
flowchart TD
    A["Text input\nchunk[0] — first 10,000 chars"]

    A --> B["Check _stored_values_cache\nDoes collection have existing data?"]

    B --> C{Cache\npopulated?}

    C -->|"No (empty collection)"| D["Build prompt without grounding\nBase PROMPT_TEMPLATE only"]

    C -->|"Yes (existing data)"| E["Build grounding section\nExisting values:\n  company_name: ['apple inc.']\n  doc_type: ['10-k', '10-q']"]

    E --> F["Inject grounding BEFORE 'Document Text:'\nso LLM sees it as instruction,\nnot as content to extract from"]

    D --> G["LLM call\nprompt | llm\nchain.invoke({content: text})"]
    F --> G

    G --> H["Raw LLM response\n(may be wrapped in ```json ... ```)"]

    H --> I["_parse_json_response()\nStrip markdown fences\nFind { ... } boundaries\njson.loads()"]

    I --> J{Valid JSON?}

    J -->|No| ERR["Raise ValueError\n'LLM did not return valid JSON'"]
    J -->|Yes| K["Normalize string values\nv.lower().strip()\nfor all str fields"]

    K --> L(["metadata dict\n{company_name: 'apple inc.',\n doc_type: '10-k',\n fiscal_quarter: None,\n fiscal_year: [2025]}"])
```

---

## Prompt Structure

The LLM receives a structured prompt. Grounding is injected **before** the document content section so it acts as an instruction, not data:

```
┌─────────────────────────────────────────────────────┐
│  SYSTEM INSTRUCTION                                 │
│  "You are a financial document metadata extractor"  │
│  "Return ONLY valid JSON in this format: { ... }"   │
├─────────────────────────────────────────────────────┤
│  GROUNDING (only when collection has data)          │
│  "Existing values in the collection:                │
│     company_name: ['apple inc.', 'microsoft']       │
│   Use stored value if this document is same entity" │
├─────────────────────────────────────────────────────┤
│  DOCUMENT TEXT                                      │
│  "Document Text:                                    │
│   {content — first 10,000 chars}"                   │
├─────────────────────────────────────────────────────┤
│  OUTPUT MARKER                                      │
│  "Extracted Metadata (JSON only):"                  │
│  ↑ LLM starts generating here                       │
└─────────────────────────────────────────────────────┘
```

---

## Grounding — Why It Matters

Without grounding, the same company can be stored under multiple names across ingestion runs:

```mermaid
flowchart LR
    subgraph Without Grounding
        D1["Apple 10-K 2024"] -->|"LLM extracts"| V1["company_name: 'apple'"]
        D2["Apple 10-K 2025"] -->|"LLM extracts"| V2["company_name: 'apple inc.'"]
        V1 & V2 --> Problem["Filter 'apple' misses 2025 doc\nFilter 'apple inc.' misses 2024 doc"]
    end

    subgraph With Grounding
        D3["Apple 10-K 2024"] -->|"First doc — no stored values"| V3["company_name: 'apple'"]
        V3 -->|"stored in collection"| Stored["Stored: ['apple']"]
        D4["Apple 10-K 2025"] -->|"Sees stored: ['apple']"| V4["company_name: 'apple'\n(reuses stored value)"]
        V3 & V4 --> Fixed["Filter 'apple' finds both docs"]
    end
```

---

## Custom Metadata via YAML

By default, RAGWire extracts 4 financial fields. You can define any fields via `metadata.yaml`:

```mermaid
flowchart TD
    YAML["metadata.yaml\nfields:\n  - name: department\n  - name: author\n  - name: document_date"]

    YAML --> BP["build_prompt_from_fields()\nAuto-builds extraction prompt\nfrom field definitions"]

    BP --> PT["Custom prompt template\nwith {content} placeholder"]

    PT --> ME["MetadataExtractor(llm, prompt_template)\nSame extraction logic\nDifferent fields"]
```

The `from_yaml()` classmethod handles this automatically — no code change needed, just point `metadata.config_file` to your YAML.

---

## JSON Parsing Robustness

LLMs sometimes wrap JSON in markdown code fences. The parser handles all variants:

```
Input variants handled:
  ```json              → stripped
  { ... }              ```
  ```                  → stripped
  { ... }              ```
  Some text { ... }    → find first { and last }
  { ... }              → used as-is
```
