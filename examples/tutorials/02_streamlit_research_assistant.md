# Video 02 — Multi-Document Research Assistant with Streamlit

**Framework**: Streamlit
**Application**: Research paper explorer with custom metadata, MMR diversity retrieval, and citation export
**Difficulty**: Beginner–Intermediate

---

## What You'll Build

A Streamlit research assistant that:
- Ingests academic papers using a custom metadata schema (title, authors, year, research_focus)
- Lets users explore documents by research topic, author, or year
- Uses MMR retrieval for diverse, non-redundant results
- Exports a citation list from retrieved sources

---

## Install

```bash
pip install ragwire streamlit
```

---

## Project Structure

```
research_assistant/
├── app.py
├── config.yaml
└── metadata.yaml   # custom schema for research papers
```

---

## Custom Metadata Schema (`metadata.yaml`)

```yaml
fields:
  - name: title
    type: string
    description: "Full title of the paper"

  - name: authors
    type: list
    description: "List of author names"

  - name: publication_year
    type: integer
    description: "Year the paper was published"

  - name: research_focus
    type: list
    description: "Main research topics or keywords"
```

---

## Code: `app.py`

```python
import streamlit as st
from ragwire import RAGWire, Config

st.set_page_config(page_title="Research Assistant", layout="wide")
st.title("Research Paper Assistant")

@st.cache_resource
def load_pipeline():
    config = Config("config.yaml")
    return RAGWire(config)

pipeline = load_pipeline()

# Sidebar — metadata exploration
with st.sidebar:
    st.header("Explore Collection")

    if st.button("Discover Fields"):
        fields = pipeline.discover_metadata_fields()
        st.session_state["fields"] = fields
        st.json(fields)

    if st.button("Get Topics"):
        values = pipeline.get_field_values(["research_focus"])
        st.session_state["topics"] = values.get("research_focus", [])

    topics = st.session_state.get("topics", [])
    selected_topics = st.multiselect("Filter by topic", topics)
    year_range = st.slider("Publication year", 2000, 2025, (2020, 2025))

# Main — chat + results
query = st.text_input("Search research papers...", placeholder="e.g. transformer attention mechanisms")

if query:
    filters = {}
    if selected_topics:
        filters["research_focus"] = selected_topics
    filters["publication_year"] = list(range(year_range[0], year_range[1] + 1))

    with st.spinner("Finding relevant papers..."):
        # Use MMR for diverse results
        from ragwire import get_retriever
        qdrant_store = pipeline.vector_store
        retriever = get_retriever(qdrant_store, search_type="mmr", top_k=8, fetch_k=20)
        docs = retriever.invoke(query)

    st.subheader(f"Found {len(docs)} relevant chunks")

    # Group by paper
    papers = {}
    for doc in docs:
        title = doc.metadata.get("title", doc.metadata.get("file_name", "Unknown"))
        if title not in papers:
            papers[title] = {"meta": doc.metadata, "chunks": []}
        papers[title]["chunks"].append(doc.page_content)

    for title, data in papers.items():
        meta = data["meta"]
        with st.expander(f"**{title}**"):
            st.markdown(f"- **Authors**: {', '.join(meta.get('authors', []))}")
            st.markdown(f"- **Year**: {meta.get('publication_year', 'N/A')}")
            st.markdown(f"- **Topics**: {', '.join(meta.get('research_focus', []))}")
            for chunk in data["chunks"]:
                st.text(chunk[:400] + "...")

    # Citation export
    if st.button("Export Citations (BibTeX)"):
        citations = []
        for title, data in papers.items():
            meta = data["meta"]
            authors = " and ".join(meta.get("authors", ["Unknown"]))
            year = meta.get("publication_year", "0000")
            key = f"{authors.split()[0].lower()}{year}"
            citations.append(
                f"@article{{{key},\n  title={{{title}}},\n  author={{{authors}}},\n  year={{{year}}}\n}}"
            )
        st.code("\n\n".join(citations), language="bibtex")
```

---

## Key RAGWire Concepts Covered

| Concept | Code location |
|---------|---------------|
| Custom `metadata.yaml` | Domain-specific field extraction |
| `discover_metadata_fields()` | Dynamic schema discovery |
| `get_field_values()` | Faceted filtering UI population |
| `get_retriever(..., search_type="mmr")` | Diversity-optimized retrieval |

---

## What to Explain in Video

1. Why custom metadata schemas matter for research (3 min)
2. Writing `metadata.yaml` with types and descriptions (5 min)
3. MMR vs similarity vs hybrid — when to use each (7 min)
4. `discover_metadata_fields()` and `get_field_values()` for dynamic UIs (5 min)
5. Building multi-select filters in Streamlit (5 min)
6. Citation export feature (5 min)
7. Live demo with research papers (5 min)
