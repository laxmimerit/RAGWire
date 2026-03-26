# Metadata Schema Examples

Ready-to-use `metadata.yaml` files for common domains. Copy the one that fits your use case, save it, and point `config.yaml` to it:

```yaml
metadata:
  config_file: "your_metadata.yaml"
```

---

## SEC / Financial Filings

```yaml
prompt: |
  You are parsing an SEC filing. This document IS an SEC filing — treat it as such.
  Extract the four fields below. All values must be lowercase strings.

  **company_name**: The company that filed this document.
  Scan for "registrant", the title block, or the company name printed above the form number.
  Use the full legal name. Example: "AMAZON.COM, INC." → 'amazon.com inc.'

  **doc_type**: The SEC form type. Map exactly:
  "Form 10-K" or "Annual Report on Form 10-K" → '10-k'
  "Form 10-Q" or "Quarterly Report on Form 10-Q" → '10-q'
  "Form 8-K" or "Current Report on Form 8-K" → '8-k'

  **fiscal_year**: The primary year this filing covers.
  Look for "fiscal year ended", "year ended", "for the year ended".
  Return a single 4-digit integer. Example: "Year ended December 31, 2024" → 2024

  **fiscal_quarter**: The quarter this filing covers. Only for 10-Q filings — null for 10-K and 8-K.
  Look for "quarter ended", "three months ended", or "Q1/Q2/Q3".
  Map: first/Q1 → 'q1', second/Q2 → 'q2', third/Q3 → 'q3', fourth/Q4 → 'q4'

  ## Grounding
  If existing collection values are provided above, reuse the exact stored value when this document refers to the same entity.

fields:
  - name: company_name
    description: "Full legal registrant name in lowercase. Example: 'alphabet inc.', 'apple inc.'"

  - name: doc_type
    description: "SEC form type: '10-k', '10-q', or '8-k'. Null if not an SEC filing."
    values: ["10-k", "10-q", "8-k"]

  - name: fiscal_year
    description: "Primary fiscal year as a 4-digit integer (e.g. 2024). Null if not stated."
    type: integer

  - name: fiscal_quarter
    description: "Fiscal quarter: 'q1', 'q2', 'q3', or 'q4'. Only for 10-Q. Null for 10-K and 8-K."
    values: ["q1", "q2", "q3", "q4"]
```

---

## Legal / Contracts

```yaml
prompt: |
  You are a legal document analyst. Extract structured metadata from the contract or legal document below.
  The document may be a contract, agreement, policy, NDA, or court filing.

  **organization**: The primary organization or company that authored or is bound by this document.
  Use the full legal name in lowercase. Null if no organization is identifiable.

  **doc_type**: The type of legal document. Map to the closest match:
  Contract, Service Agreement, Master Agreement → 'contract'
  Non-Disclosure Agreement, NDA, Confidentiality Agreement → 'nda'
  Policy, Code of Conduct, Terms of Service → 'policy'
  Court Filing, Complaint, Motion, Order → 'court-filing'
  Amendment, Addendum → 'amendment'
  Null if none fit.

  **effective_year**: The year this document comes into effect or was signed.
  Look for "effective date", "dated as of", "executed on", or "signed on".
  Return a single 4-digit integer. Null if not stated.

  **jurisdiction**: The governing law or jurisdiction stated in the document.
  Look for "governed by the laws of", "jurisdiction of", or "applicable law".
  Return in lowercase (e.g. 'new york', 'california', 'england and wales'). Null if not stated.

  **parties**: All named parties to the document as a list in lowercase.
  Look for "between", "by and between", or signature blocks.

  ## Rules
  - All string values must be lowercase.
  - Null if a field is not clearly stated.

fields:
  - name: organization
    description: "Primary organization or company in lowercase. Example: 'acme corp', 'google llc'"

  - name: doc_type
    description: "Type of legal document in lowercase."
    values: ["contract", "nda", "policy", "court-filing", "amendment"]

  - name: effective_year
    description: "Year the document takes effect as a 4-digit integer. Null if not stated."
    type: integer

  - name: jurisdiction
    description: "Governing law or jurisdiction in lowercase (e.g. 'new york', 'california'). Null if not stated."

  - name: parties
    description: "List of all named parties to the document in lowercase."
    type: list
```

---

## Healthcare / Medical

```yaml
prompt: |
  You are a medical document analyst. Extract structured metadata from the clinical or medical document below.
  The document may be a clinical study, guideline, patient report, or medical literature.

  **condition**: The primary medical condition(s) or disease(s) discussed.
  Use lowercase-hyphenated format. Example: "Type 2 Diabetes" → 'type-2-diabetes'.

  **specialty**: The medical specialty this document falls under.
  Map to the closest match from the allowed values. Null if unclear.

  **doc_type**: The type of medical document. Map to the closest:
  Randomized Controlled Trial, RCT → 'rct'
  Systematic Review, Meta-Analysis → 'systematic-review'
  Case Study, Case Report → 'case-study'
  Clinical Guideline, Practice Guideline → 'guideline'
  Observational Study, Cohort Study → 'cohort-study'

  **publication_year**: The year this document was published or last updated.
  Look for a publication date, "published", "revised", or copyright year.
  Return a single 4-digit integer. Null if not stated.

  **treatment**: The primary treatment(s) or intervention(s) evaluated or described.
  Use lowercase. Return as a list if multiple treatments are discussed.

  ## Rules
  - All string values must be lowercase and hyphenated where appropriate.
  - Null if a field is not clearly present.

fields:
  - name: condition
    description: "Primary medical condition(s) in lowercase-hyphenated format (e.g. 'type-2-diabetes', 'heart-failure')."
    type: list

  - name: specialty
    description: "Medical specialty in lowercase."
    values: ["cardiology", "oncology", "neurology", "orthopedics", "psychiatry", "endocrinology", "general-practice"]

  - name: doc_type
    description: "Type of medical document."
    values: ["rct", "systematic-review", "case-study", "guideline", "cohort-study"]

  - name: publication_year
    description: "Year of publication as a 4-digit integer. Null if not stated."
    type: integer

  - name: treatment
    description: "Primary treatment(s) or intervention(s) discussed in lowercase."
    type: list
```

---

## Academic / Research Papers

```yaml
prompt: |
  You are an academic document analyst. Extract structured metadata from the research paper below.
  Focus on the title, authors, and key research characteristics.

  **title**: The full title of the paper exactly as it appears. Do not paraphrase.

  **authors**: All author names as they appear in the paper.
  Return as a list. Example: ["john a. smith", "jane doe"]

  **publication_year**: The year the paper was published or last revised.
  Look for a publication date, journal issue, or copyright line.
  Return a single 4-digit integer.

  **domain**: The primary academic domain or field of study in lowercase.
  Map to the closest match from the allowed values.

  **keywords**: Key topics or concepts from the paper in lowercase-hyphenated format.
  Extract from the abstract, keywords section, or infer from the title. Return as a list.

  ## Rules
  - All string values must be lowercase.
  - Null if a field is genuinely absent.

fields:
  - name: title
    description: "Full title of the paper exactly as stated. Do not paraphrase."

  - name: authors
    description: "List of all author names in lowercase as they appear in the paper."
    type: list

  - name: publication_year
    description: "Year of publication as a 4-digit integer. Null if not found."
    type: integer

  - name: domain
    description: "Primary academic domain in lowercase."
    values: ["computer-science", "medicine", "biology", "physics", "economics", "psychology", "engineering", "mathematics"]

  - name: keywords
    description: "Key topics and concepts in lowercase-hyphenated format."
    type: list
```

---

## HR / Internal Documents

```yaml
prompt: |
  You are an HR document analyst. Extract structured metadata from the internal business document below.
  The document may be a policy, job description, performance review, or employee handbook section.

  **department**: The department or business unit this document applies to.
  Use lowercase. Example: "Human Resources" → 'human-resources', "Engineering" → 'engineering'.
  Null if the document applies company-wide.

  **doc_type**: The type of HR document. Map to the closest:
  Policy, Code of Conduct, Handbook → 'policy'
  Job Description, Job Posting → 'job-description'
  Performance Review, Appraisal → 'performance-review'
  Training Material, Onboarding Guide → 'training'
  Compensation, Benefits Guide → 'compensation'

  **effective_year**: The year this document is effective or was last revised.
  Look for "effective date", "last updated", "revised", or "version date".
  Return a single 4-digit integer. Null if not stated.

  **audience**: Who this document is intended for in lowercase.
  Example: 'all employees', 'managers', 'new hires', 'executives'. Null if not stated.

  ## Rules
  - All string values must be lowercase and hyphenated where appropriate.
  - Null if a field is not clearly stated.

fields:
  - name: department
    description: "Department or business unit in lowercase-hyphenated format (e.g. 'engineering', 'human-resources'). Null if company-wide."

  - name: doc_type
    description: "Type of HR document in lowercase."
    values: ["policy", "job-description", "performance-review", "training", "compensation"]

  - name: effective_year
    description: "Year the document is effective or was last revised as a 4-digit integer. Null if not stated."
    type: integer

  - name: audience
    description: "Intended audience in lowercase (e.g. 'all employees', 'managers'). Null if not stated."
```

---

## Auto-Filter with Custom Schemas

When you configure a custom `metadata.yaml`, RAGWire automatically derives its **filter fields** from the field names in that file. `auto_filter` then extracts values for exactly those fields — not the default financial ones.

### How it works

```
metadata.yaml fields → rag.filter_fields → auto_filter extracts → retrieve() narrows results
```

- **Default schema** (no `metadata.yaml`): filter fields are `company_name`, `doc_type`, `fiscal_quarter`, `fiscal_year`
- **Custom schema**: filter fields become whatever `name:` entries you defined in your YAML

### Check which fields are active

```python
rag = RAGWire(Config("config.yaml"))
print(rag.filter_fields)
# Legal schema  → ['organization', 'doc_type', 'effective_year', 'jurisdiction', 'parties']
# Academic schema → ['title', 'authors', 'publication_year', 'domain', 'keywords']
# HR schema     → ['department', 'doc_type', 'effective_year', 'audience']
```

### Enable auto_filter in config.yaml

```yaml
metadata:
  config_file: "your_schema.yaml"   # custom schema

retriever:
  auto_filter: true    # LLM extracts filters from every query automatically
```

With this set, any call to `retrieve()` with no explicit filters will automatically extract metadata filters using your schema's fields:

```python
# Legal schema — auto_filter extracts jurisdiction and doc_type from the query
results = rag.retrieve("NDA agreements governed by California law")
# → internally applies: {"doc_type": "nda", "jurisdiction": "california"}

# Academic schema — extracts domain and publication_year
results = rag.retrieve("computer science papers from 2023")
# → internally applies: {"domain": "computer-science", "publication_year": 2023}

# HR schema — extracts department and doc_type
results = rag.retrieve("engineering job descriptions")
# → internally applies: {"department": "engineering", "doc_type": "job-description"}
```

### auto_filter vs explicit filters vs agent-controlled

| Mode | Config | When to use |
|------|--------|-------------|
| `auto_filter: true` | `retriever.auto_filter: true` | Simple chatbots and search UIs — one-liner retrieval |
| Explicit filters | `auto_filter: false` (default) + pass `filters=` | Programmatic pipelines where you know the inputs |
| Agent-controlled | `auto_filter: false` + call `extract_filters()` or `get_filter_context()` | Agents that need to inspect or adjust filters before retrieving |

!!! note "auto_filter is off by default"
    Set `auto_filter: true` only when you want every `retrieve()` call to run an LLM filter-extraction step. This adds latency. For agents, keep the default and use `extract_filters()` or `get_filter_context()` so the agent retains full control.

### Stored-value matching

When `auto_filter` runs, it reads the actual values already stored in your Qdrant collection and feeds them to the LLM. This allows intelligent alias resolution:

```
Query: "Google filings"
Stored company_name values: ["alphabet inc."]
→ auto_filter extracts: {"company_name": "alphabet inc."}   ✓ correct match

Query: "2023 oncology studies"
Stored specialty values: ["cardiology", "oncology", "neurology"]
→ auto_filter extracts: {"specialty": "oncology", "publication_year": 2023}   ✓
```

The same matching logic applies to all custom schemas — stored values anchor the extraction so results are consistent as the collection grows.
