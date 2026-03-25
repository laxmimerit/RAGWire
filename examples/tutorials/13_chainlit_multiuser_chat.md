# Video 13 — Multi-User Document Chat with Chainlit

**Framework**: Chainlit
**Application**: Authenticated multi-user chat where each user has isolated document collections and conversation history
**Difficulty**: Intermediate–Advanced

---

## What You'll Build

A production-style Chainlit app with:
- Username/password authentication (custom auth callback)
- Per-user isolated Qdrant collections (each user's docs stay private)
- Persistent conversation history via Chainlit's data layer
- Admin user who can see all collections
- Document upload + per-user ingestion
- Session-aware streaming responses

---

## Install

```bash
pip install ragwire chainlit langchain-ollama
```

---

## Project Structure

```
multiuser_chat/
├── app.py
├── config.yaml
├── .chainlit/
│   └── config.toml    # enable auth
└── users.json         # simple user store (use DB in production)
```

---

## `.chainlit/config.toml`

```toml
[project]
enable_telemetry = false

[features]
multi_modal = true

[auth]
# Custom auth — implemented in app.py via @cl.password_auth_callback
```

---

## `users.json`

```json
{
  "alice": {"password": "alice123", "role": "user"},
  "bob":   {"password": "bob456",   "role": "user"},
  "admin": {"password": "admin999", "role": "admin"}
}
```

---

## Code: `app.py`

```python
import chainlit as cl
from chainlit.types import ThreadDict
from ragwire import RAGWire, Config
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import json
import os
import tempfile
import yaml

# --- Auth ---

def load_users() -> dict:
    with open("users.json") as f:
        return json.load(f)

@cl.password_auth_callback
def auth_callback(username: str, password: str):
    users = load_users()
    if username in users and users[username]["password"] == password:
        return cl.User(
            identifier=username,
            metadata={"role": users[username]["role"]}
        )
    return None

# --- Per-user RAGWire pipeline factory ---

def get_user_pipeline(username: str) -> RAGWire:
    """
    Create a RAGWire instance with a user-specific Qdrant collection.
    Each user's documents are isolated in their own collection.
    """
    # Load base config
    with open("config.yaml") as f:
        cfg_data = yaml.safe_load(f)

    # Override collection name per user
    cfg_data["vectorstore"]["collection_name"] = f"user_{username}_docs"

    # Write temp config
    tmp_path = f"/tmp/config_{username}.yaml"
    with open(tmp_path, "w") as f:
        yaml.dump(cfg_data, f)

    return RAGWire(Config(tmp_path))

# --- Chat Session ---

@cl.on_chat_start
async def on_start():
    user = cl.user_session.get("user")
    username = user.identifier
    role = user.metadata.get("role", "user")

    # Initialize per-user pipeline
    pipeline = get_user_pipeline(username)
    llm = ChatOllama(model="qwen2.5:7b", streaming=True)

    cl.user_session.set("pipeline", pipeline)
    cl.user_session.set("llm", llm)
    cl.user_session.set("username", username)
    cl.user_session.set("role", role)

    greeting = f"Hello **{username}**! "
    if role == "admin":
        greeting += "You have admin access — you can view collection stats."
    else:
        greeting += "Upload documents or ask questions about your files."

    await cl.Message(content=greeting).send()

    # Show user's document count if any
    try:
        fields = pipeline.discover_metadata_fields()
        if fields:
            await cl.Message(content=f"Your collection has documents with fields: `{list(fields.keys())}`").send()
    except Exception:
        pass

    # Offer file upload on start (Chainlit v2 AskFileMessage — preferred pattern)
    if role != "admin":
        files = await cl.AskFileMessage(
            content="Upload documents to your private collection (optional):",
            accept=["application/pdf", "text/plain", "text/markdown",
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"],
            max_files=10,
            timeout=30,
            raise_on_timeout=False
        ).send()
        if files:
            await handle_file_upload(files, pipeline, username)

@cl.on_message
async def on_message(message: cl.Message):
    pipeline: RAGWire = cl.user_session.get("pipeline")
    llm: ChatOllama = cl.user_session.get("llm")
    username: str = cl.user_session.get("username")
    role: str = cl.user_session.get("role")

    # Handle admin commands
    if role == "admin" and message.content.startswith("/"):
        await handle_admin_command(message.content, pipeline)
        return

    # Handle file uploads
    if message.elements:
        await handle_file_upload(message.elements, pipeline, username)
        return

    # Regular Q&A
    docs = pipeline.retrieve(message.content, top_k=4)

    if not docs:
        await cl.Message(
            content="I couldn't find relevant information in your documents. Try uploading some files first!"
        ).send()
        return

    context = "\n\n---\n\n".join(d.page_content for d in docs)

    # Get conversation history from session
    history = cl.user_session.get("history", [])
    messages = [SystemMessage(content=(
        "You are a helpful assistant. Answer questions using only the provided document context. "
        "If you don't know, say so honestly."
    ))]
    for turn in history[-4:]:
        messages.append(HumanMessage(content=turn["user"]))
        messages.append(AIMessage(content=turn["assistant"]))
    messages.append(HumanMessage(content=f"Context:\n{context}\n\nQuestion: {message.content}"))

    # Stream response
    sources = [
        cl.Text(
            name=f"[{i+1}] {d.metadata.get('file_name', 'doc')}",
            content=d.page_content,
            display="side"
        )
        for i, d in enumerate(docs)
    ]
    response_msg = cl.Message(content="", elements=sources)
    await response_msg.send()

    full_response = ""
    async for chunk in llm.astream(messages):
        full_response += chunk.content
        await response_msg.stream_token(chunk.content)
    await response_msg.update()

    # Save to history
    history.append({"user": message.content, "assistant": full_response})
    cl.user_session.set("history", history[-10:])  # keep last 10 turns


async def handle_file_upload(elements, pipeline: RAGWire, username: str):
    """Process drag-dropped files from message.elements into user's isolated collection."""
    with tempfile.TemporaryDirectory() as tmpdir:
        for elem in elements:
            # In Chainlit v2, uploaded elements have .path and .name directly
            dest = os.path.join(tmpdir, elem.name)
            with open(elem.path, "rb") as src, open(dest, "wb") as dst:
                dst.write(src.read())

        msg = cl.Message(content=f"Ingesting files for user `{username}`...")
        await msg.send()

        stats = pipeline.ingest_directory(tmpdir)
        await msg.update(
            content=f"Ingested {stats['chunks_created']} chunks from {stats['processed']} file(s). "
                    f"({stats['skipped']} already in your collection)"
        )


async def handle_admin_command(command: str, pipeline: RAGWire):
    """Handle admin-only slash commands."""
    cmd = command.strip().lower()

    if cmd == "/stats":
        fields = pipeline.discover_metadata_fields()
        values = pipeline.get_field_values(list(fields.keys())[:3])
        await cl.Message(content=f"```json\n{json.dumps(values, indent=2)}\n```").send()

    elif cmd == "/help":
        await cl.Message(content=(
            "**Admin Commands:**\n"
            "- `/stats` — Show collection metadata statistics\n"
            "- `/help` — Show this help"
        )).send()

    else:
        await cl.Message(content=f"Unknown command: `{command}`").send()


@cl.on_chat_resume
async def on_resume(thread: ThreadDict):
    """Restore session when user reconnects."""
    user = cl.user_session.get("user")
    if user:
        pipeline = get_user_pipeline(user.identifier)
        llm = ChatOllama(model="qwen2.5:7b", streaming=True)
        cl.user_session.set("pipeline", pipeline)
        cl.user_session.set("llm", llm)
```

---

## Key Concepts Covered

| Concept | Code location |
|---------|---------------|
| `@cl.password_auth_callback` | Custom authentication |
| Per-user collection isolation | `collection_name = f"user_{username}_docs"` |
| `cl.User` with metadata | Role-based access (admin vs user) |
| `@cl.on_chat_resume` | Restore session on reconnect |
| Admin slash commands | `handle_admin_command` |

---

## What to Explain in Video

1. Chainlit authentication callbacks (5 min)
2. Per-user collection isolation strategy (7 min)
3. `cl.user_session` vs server-side state (5 min)
4. `@cl.on_chat_resume` for session persistence (5 min)
5. Role-based feature gating (admin vs user) (3 min)
6. Scaling considerations: one collection per user pros/cons (5 min)
7. Live demo with two users (5 min)
