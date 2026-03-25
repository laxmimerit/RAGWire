# Video 03 — Conversational RAG Chatbot with Chainlit

**Framework**: Chainlit
**Application**: Streaming document chatbot with conversation history, source citations, and file upload
**Difficulty**: Beginner–Intermediate

---

## What You'll Build

A Chainlit chatbot that:
- Accepts document uploads directly in the chat UI
- Streams LLM responses token by token
- Shows cited source chunks as expandable elements
- Maintains full conversation history across turns
- Displays ingestion progress in the UI

---

## Install

```bash
pip install ragwire chainlit langchain-ollama
```

---

## Project Structure

```
chainlit_chatbot/
├── app.py
├── config.yaml
└── .chainlit/
    └── config.toml
```

---

## Code: `app.py`

```python
import chainlit as cl
from ragwire import RAGWire, Config
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import tempfile, os

# --- Helpers ---

async def _ingest_files(files: list, pipeline) -> None:
    """Ingest a list of AskFileResponse objects."""
    with tempfile.TemporaryDirectory() as tmpdir:
        for f in files:
            dest = os.path.join(tmpdir, f.name)
            with open(f.path, "rb") as src, open(dest, "wb") as dst:
                dst.write(src.read())
        await _ingest_directory(tmpdir, pipeline)

async def _ingest_directory(tmpdir: str, pipeline) -> None:
    msg = cl.Message(content="Ingesting documents...")
    await msg.send()
    stats = pipeline.ingest_directory(tmpdir)
    await msg.update(
        content=f"Ingested {stats['chunks_created']} chunks from "
                f"{stats['processed']} files. "
                f"({stats['skipped']} skipped as duplicates)"
    )

# ---

SYSTEM_PROMPT = """You are a helpful document assistant.
Answer questions using only the provided context.
If the answer is not in the context, say so honestly.
Be concise and cite specific details from the documents."""

@cl.on_chat_start
async def on_start():
    config = Config("config.yaml")
    pipeline = RAGWire(config)
    llm = ChatOllama(model="qwen2.5:7b", streaming=True)

    cl.user_session.set("pipeline", pipeline)
    cl.user_session.set("llm", llm)
    cl.user_session.set("history", [])

    await cl.Message(
        content="Hello! Upload documents or ask questions about already-ingested files."
    ).send()

    # Proactively ask for file upload on session start (Chainlit v2 preferred pattern)
    files = await cl.AskFileMessage(
        content="Upload PDF/DOCX files to get started (optional — skip by typing a question):",
        accept=["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "text/plain", "text/markdown"],
        max_files=10,
        timeout=30,
        raise_on_timeout=False
    ).send()

    if files:
        await _ingest_files(files, pipeline)

@cl.on_message
async def on_message(message: cl.Message):
    pipeline: RAGWire = cl.user_session.get("pipeline")
    llm: ChatOllama = cl.user_session.get("llm")
    history: list = cl.user_session.get("history")

    # Handle drag-drop file uploads in message
    if message.elements:
        with tempfile.TemporaryDirectory() as tmpdir:
            for elem in message.elements:
                dest = os.path.join(tmpdir, elem.name)
                with open(elem.path, "rb") as src, open(dest, "wb") as dst:
                    dst.write(src.read())
            await _ingest_directory(tmpdir, pipeline)
        return

    # Retrieve relevant context
    docs = pipeline.retrieve(message.content)

    if not docs:
        await cl.Message(content="I couldn't find relevant information in the documents.").send()
        return

    # Show sources as elements
    sources = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        label = f"[{i}] {meta.get('file_name', 'source')} — {meta.get('company_name', '')}"
        sources.append(cl.Text(name=label, content=doc.page_content, display="side"))

    context = "\n\n---\n\n".join(d.page_content for d in docs)

    # Build message history
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    for turn in history[-6:]:  # last 3 turns
        messages.append(HumanMessage(content=turn["user"]))
        messages.append(AIMessage(content=turn["assistant"]))
    messages.append(HumanMessage(content=f"Context:\n{context}\n\nQuestion: {message.content}"))

    # Stream response
    response_msg = cl.Message(content="", elements=sources)
    await response_msg.send()

    full_response = ""
    async for chunk in llm.astream(messages):
        token = chunk.content
        full_response += token
        await response_msg.stream_token(token)

    await response_msg.update()

    # Update history
    history.append({"user": message.content, "assistant": full_response})
    cl.user_session.set("history", history)
```

---

## Key RAGWire Concepts Covered

| Concept | Code location |
|---------|---------------|
| `ingest_directory()` with stats | File upload + real-time progress |
| `retrieve()` | Hybrid search per user message |
| `stats['skipped']` | Deduplication transparency |
| Multi-turn history | Conversation memory over documents |

---

## Chainlit Features Highlighted

| Feature | Purpose |
|---------|---------|
| `cl.on_chat_start` | Initialize pipeline per session |
| `cl.user_session` | Per-user isolated state |
| `cl.Message.stream_token()` | Token-by-token streaming |
| `cl.AskFileMessage` | Proactive file upload prompt (v2 preferred) |
| `cl.Text(..., display="side")` | Source citations panel |

---

## What to Explain in Video

1. Chainlit vs Streamlit — when to use each (3 min)
2. `cl.user_session` for per-user RAGWire instances (5 min)
3. File upload handling and temp directory pattern (5 min)
4. Async streaming with `llm.astream()` (5 min)
5. Source citations as side panel elements (5 min)
6. Conversation history truncation strategy (3 min)
7. Live demo (9 min)
