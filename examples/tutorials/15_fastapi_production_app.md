# Video 15 — Production Full-Stack RAG App with FastAPI + Streamlit

**Framework**: FastAPI (backend) + Streamlit (frontend)
**Application**: Production-grade RAG system with REST API backend, async ingestion queue, and Streamlit frontend
**Difficulty**: Advanced

---

## What You'll Build

A production-ready 2-tier architecture:

**Backend (FastAPI)**:
- `POST /ingest` — async document ingestion with job tracking
- `GET /ingest/{job_id}` — check ingestion status
- `POST /query` — query with optional filters, returns sources + answer
- `GET /collections` — list available collections and stats
- `DELETE /collection/{name}` — remove a collection
- Background task queue for ingestion jobs

**Frontend (Streamlit)**:
- Connects to the API backend
- File upload triggers async ingestion job
- Polls for job completion
- Chat interface with source attribution
- Collection management panel

---

## Install

```bash
pip install ragwire fastapi uvicorn streamlit httpx python-multipart
```

---

## Project Structure

```
production_app/
├── backend/
│   ├── main.py          # FastAPI app
│   ├── models.py        # Pydantic request/response models
│   └── config.yaml
├── frontend/
│   └── app.py           # Streamlit frontend
└── docker-compose.yml   # Optional: containerized setup
```

---

## Backend: `backend/models.py`

```python
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from enum import Enum

class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"

class IngestJob(BaseModel):
    job_id: str
    status: JobStatus
    files: List[str]
    stats: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class QueryRequest(BaseModel):
    query: str
    filters: Optional[Dict[str, Any]] = None
    top_k: int = 5
    include_sources: bool = True

class SourceDoc(BaseModel):
    file_name: str
    content_preview: str
    metadata: Dict[str, Any]
    score: Optional[float] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDoc]
    query: str
    filters_applied: Optional[Dict[str, Any]] = None

class CollectionInfo(BaseModel):
    name: str
    document_count: int
    metadata_fields: List[str]
```

---

## Backend: `backend/main.py`

```python
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ragwire import RAGWire, Config
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from models import IngestJob, QueryRequest, QueryResponse, SourceDoc, JobStatus, CollectionInfo
import uuid
import tempfile
import os
import shutil
from typing import List

app = FastAPI(
    title="RAGWire API",
    description="Production RAG API backed by RAGWire",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# --- App State ---

pipeline = RAGWire(Config("config.yaml"))
llm = ChatOllama(model="qwen2.5:7b")
jobs: dict[str, IngestJob] = {}  # In production: use Redis

# --- Background Tasks ---

def run_ingestion(job_id: str, directory: str, file_names: List[str]):
    """Background ingestion task."""
    try:
        jobs[job_id].status = JobStatus.RUNNING
        stats = pipeline.ingest_directory(directory, recursive=False)
        jobs[job_id].status = JobStatus.DONE
        jobs[job_id].stats = stats
    except Exception as e:
        jobs[job_id].status = JobStatus.FAILED
        jobs[job_id].error = str(e)
    finally:
        shutil.rmtree(directory, ignore_errors=True)

# --- Routes ---

@app.post("/ingest", response_model=IngestJob, status_code=202)
async def ingest_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...)
):
    """Upload and asynchronously ingest documents."""
    job_id = f"job_{uuid.uuid4().hex[:8]}"
    tmpdir = tempfile.mkdtemp(prefix=f"ragwire_{job_id}_")

    file_names = []
    for upload in files:
        dest = os.path.join(tmpdir, upload.filename)
        with open(dest, "wb") as f:
            content = await upload.read()
            f.write(content)
        file_names.append(upload.filename)

    job = IngestJob(job_id=job_id, status=JobStatus.PENDING, files=file_names)
    jobs[job_id] = job

    background_tasks.add_task(run_ingestion, job_id, tmpdir, file_names)

    return job


@app.get("/ingest/{job_id}", response_model=IngestJob)
async def get_ingestion_status(job_id: str):
    """Check the status of an ingestion job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return jobs[job_id]


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the document collection and get an LLM-generated answer."""
    docs = pipeline.retrieve(
        request.query,
        filters=request.filters,
        top_k=request.top_k
    )

    if not docs:
        return QueryResponse(
            answer="No relevant documents found for this query.",
            sources=[],
            query=request.query,
            filters_applied=request.filters
        )

    context = "\n\n---\n\n".join(d.page_content for d in docs)
    messages = [
        SystemMessage(content="Answer using only the provided context. Be precise and cite specifics."),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {request.query}")
    ]
    result = llm.invoke(messages)

    sources = []
    if request.include_sources:
        sources = [
            SourceDoc(
                file_name=d.metadata.get("file_name", "unknown"),
                content_preview=d.page_content[:250],
                metadata={k: v for k, v in d.metadata.items()
                          if k not in ("text", "page_content")}
            )
            for d in docs
        ]

    return QueryResponse(
        answer=result.content,
        sources=sources,
        query=request.query,
        filters_applied=request.filters
    )


@app.get("/collections/info", response_model=CollectionInfo)
async def get_collection_info():
    """Get metadata about the current collection."""
    try:
        fields = pipeline.discover_metadata_fields()
        return CollectionInfo(
            name=pipeline.vector_store.collection_name,
            document_count=-1,  # implement with Qdrant count API
            metadata_fields=list(fields.keys())
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/filters", response_model=dict)
async def get_filter_options(query: str = ""):
    """Get available filter options, optionally for a specific query."""
    if query:
        context = pipeline.get_filter_context(query)
        return {"filter_context": context}
    else:
        fields = pipeline.discover_metadata_fields()
        values = pipeline.get_field_values(list(fields.keys()))
        return {"fields": fields, "values": values}


@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}
```

---

## Frontend: `frontend/app.py`

```python
import streamlit as st
import httpx
import time

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="RAGWire App", layout="wide")
st.title("RAGWire Document Intelligence")

# Sidebar — upload
with st.sidebar:
    st.header("Upload Documents")
    files = st.file_uploader("Select files", accept_multiple_files=True, type=["pdf", "docx", "txt"])

    if st.button("Upload & Ingest") and files:
        with httpx.Client() as client:
            response = client.post(
                f"{API_BASE}/ingest",
                files=[("files", (f.name, f.read(), "application/octet-stream")) for f in files]
            )
        if response.status_code == 202:
            job = response.json()
            st.session_state["job_id"] = job["job_id"]
            st.info(f"Job started: `{job['job_id']}`")
        else:
            st.error(f"Upload failed: {response.text}")

    # Poll job status
    if "job_id" in st.session_state:
        job_id = st.session_state["job_id"]
        if st.button("Check Status"):
            with httpx.Client() as client:
                resp = client.get(f"{API_BASE}/ingest/{job_id}")
            job = resp.json()
            status = job["status"]
            if status == "done":
                st.success(f"Done! {job['stats']}")
                del st.session_state["job_id"]
            elif status == "failed":
                st.error(f"Failed: {job.get('error')}")
            else:
                st.info(f"Status: {status}")

    st.header("Filters")
    if st.button("Load Filter Options"):
        with httpx.Client() as client:
            resp = client.get(f"{API_BASE}/filters")
        st.session_state["filter_data"] = resp.json()

    filter_data = st.session_state.get("filter_data", {})
    selected_filters = {}
    for field, values in filter_data.get("values", {}).items():
        if values:
            selected = st.multiselect(field, values, key=f"filter_{field}")
            if selected:
                selected_filters[field] = selected

# Main — chat
query = st.chat_input("Ask a question...")
if query:
    with st.chat_message("user"):
        st.write(query)

    with st.spinner("Querying..."):
        with httpx.Client(timeout=60) as client:
            response = client.post(
                f"{API_BASE}/query",
                json={"query": query, "filters": selected_filters or None, "top_k": 5}
            )

    with st.chat_message("assistant"):
        if response.status_code == 200:
            data = response.json()
            st.write(data["answer"])

            if data.get("filters_applied"):
                st.caption(f"Filters: {data['filters_applied']}")

            with st.expander(f"Sources ({len(data['sources'])})"):
                for src in data["sources"]:
                    st.markdown(f"**{src['file_name']}**")
                    st.text(src["content_preview"])
                    st.json(src["metadata"])
        else:
            st.error(f"Query failed: {response.text}")
```

---

## Running the App

```bash
# Terminal 1 — Start backend
cd backend
uvicorn main:app --reload --port 8000

# Terminal 2 — Start frontend
cd frontend
streamlit run app.py
```

---

## Key Concepts Covered

| Concept | Code location |
|---------|---------------|
| Async background ingestion | `BackgroundTasks` + job tracking |
| Job status polling | `GET /ingest/{job_id}` endpoint |
| Decoupled frontend/backend | `httpx` client in Streamlit |
| REST API over RAGWire | All pipeline features exposed as endpoints |
| CORS middleware | `CORSMiddleware` for browser access |

---

## What to Explain in Video

1. Why decouple frontend and backend (3 min)
2. FastAPI `BackgroundTasks` for async ingestion (7 min)
3. Job status polling pattern (5 min)
4. Pydantic request/response models for API contracts (5 min)
5. `httpx` vs `requests` in async apps (3 min)
6. Docker Compose setup for production (5 min)
7. API documentation at `/docs` (Swagger UI) (2 min)
8. Live demo (10 min)

---

## `docker-compose.yml` (Bonus)

```yaml
version: "3.9"
services:
  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

  ollama:
    image: ollama/ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama

  backend:
    build: ./backend
    ports:
      - "8000:8000"
    depends_on:
      - qdrant
      - ollama
    environment:
      - QDRANT_URL=http://qdrant:6333
      - OLLAMA_URL=http://ollama:11434

  frontend:
    build: ./frontend
    ports:
      - "8501:8501"
    depends_on:
      - backend

volumes:
  qdrant_data:
  ollama_data:
```
