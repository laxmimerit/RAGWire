# Video 01 — Financial Document Q&A App with Streamlit

**Framework**: Streamlit
**Application**: Ask questions over SEC 10-K / 10-Q filings with metadata-aware retrieval
**Difficulty**: Beginner

---

## What You'll Build

A Streamlit web app where users can:
- Upload financial PDFs (10-K, 10-Q, 8-K)
- Auto-ingest documents into RAGWire
- Ask natural language questions with real-time answers
- Filter answers by company, fiscal year, or document type via sidebar controls

---

## Install

```bash
pip install ragwire streamlit
```

---

## Project Structure

```
financial_qa/
├── app.py
├── config.yaml
├── finance_metadata.yaml
└── data/
    └── (drop PDFs here)
```

---

## Code: `app.py`

```python
import streamlit as st
from ragwire import RAGWire, Config

st.set_page_config(page_title="Financial Doc Q&A", layout="wide")
st.title("Financial Document Q&A")

@st.cache_resource
def load_pipeline():
    config = Config("config.yaml")
    return RAGWire(config)

pipeline = load_pipeline()

# Sidebar — ingest
with st.sidebar:
    st.header("Ingest Documents")
    uploaded = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    if st.button("Ingest") and uploaded:
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmpdir:
            for f in uploaded:
                path = os.path.join(tmpdir, f.name)
                with open(path, "wb") as out:
                    out.write(f.read())
            stats = pipeline.ingest_directory(tmpdir)
            st.success(f"Ingested {stats['chunks_created']} chunks from {stats['processed']} files")

    st.header("Filters")
    company = st.text_input("Company name (optional)")
    year = st.number_input("Fiscal year (0 = any)", min_value=0, max_value=2100, value=0)

# Main — chat
query = st.chat_input("Ask a question about the documents...")
if query:
    filters = {}
    if company:
        filters["company_name"] = [company.lower()]
    if year:
        filters["fiscal_year"] = [year]

    with st.spinner("Searching..."):
        docs = pipeline.retrieve(query, filters=filters if filters else None)

    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        if not docs:
            st.warning("No relevant documents found.")
        else:
            # Build context and answer with LLM
            context = "\n\n---\n\n".join(d.page_content for d in docs)
            from langchain_ollama import ChatOllama
            from langchain_core.messages import HumanMessage, SystemMessage

            llm = ChatOllama(model="qwen2.5:7b")
            messages = [
                SystemMessage(content="You are a financial analyst. Answer using only the provided context."),
                HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}")
            ]
            response = llm.invoke(messages)
            st.write(response.content)

            with st.expander("Source chunks"):
                for i, doc in enumerate(docs, 1):
                    meta = doc.metadata
                    st.markdown(f"**[{i}]** `{meta.get('file_name', 'unknown')}` — "
                                f"{meta.get('company_name', '')} {meta.get('fiscal_year', '')}")
                    st.text(doc.page_content[:300] + "...")
```

---

## Key RAGWire Concepts Covered

| Concept | Code location |
|---------|---------------|
| `RAGWire(config)` | Pipeline initialization |
| `ingest_directory()` | Bulk PDF ingestion with dedup |
| `retrieve(query, filters)` | Metadata-filtered hybrid search |
| `FinancialMetadata` | Auto-extracted company, year, doc_type fields |

---

## What to Explain in Video

1. RAGWire architecture overview (5 min)
2. Setting up Qdrant with Docker (2 min)
3. `config.yaml` walkthrough (3 min)
4. `ingest_directory()` with deduplication (5 min)
5. `retrieve()` with metadata filters (5 min)
6. Building the Streamlit UI (10 min)
7. Live demo with Apple 10-K (5 min)

---

## Extensions to Mention

- Swap Ollama for OpenAI by changing `config.yaml`
- Add chart visualization of financials extracted by LLM
- Deploy to Streamlit Cloud
