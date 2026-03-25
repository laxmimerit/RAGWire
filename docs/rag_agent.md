# Building a RAG Agent

Combine RAGWire's retrieval with LangChain's `create_agent` to build a conversational Q&A agent that answers questions from your document knowledge base.

---

## How it works

1. RAGWire ingests documents and handles retrieval
2. You wrap the retriever as a `@tool` — the LLM can call it when it needs information
3. `create_agent` gives the LLM access to that tool and manages the conversation loop

---

## Prerequisites

```bash
# Core
pip install "ragwire[openai]"   # or [ollama], [groq], [anthropic]
pip install fastembed           # for hybrid search
pip install langgraph           # for memory / checkpointing
```

---

## 1. Ingest Documents

```python
from ragwire import RAGWire

rag = RAGWire("config.yaml")

# Ingest all documents from a directory
stats = rag.ingest_directory("data/")
print(f"Ingested {stats['processed']} docs, {stats['chunks_created']} chunks")
```

---

## 2. Define the Retrieval Tool

```python
from langchain.tools import tool

@tool
def search_documents(query: str) -> str:
    """Search the document knowledge base for relevant information.

    Use this whenever the user asks a question that may be answered
    by the ingested documents.
    """
    results = rag.retrieve(query, top_k=5)
    if not results:
        return "No relevant documents found."

    chunks = []
    for doc in results:
        source = doc.metadata.get("file_name", "unknown")
        company = doc.metadata.get("company_name", "")
        year = doc.metadata.get("fiscal_year", "")
        header = f"[{source}" + (f" | {company} {year}" if company else "") + "]"
        chunks.append(f"{header}\n{doc.page_content}")

    return "\n\n---\n\n".join(chunks)
```

---

## 3. Give the Agent Metadata Awareness (Optional but Recommended)

By default the agent doesn't know what companies, document types, or years are in your collection. Without this, it may hallucinate filter values or skip filtering entirely.

### Option A — `get_filter_context()` (recommended for agents)

The simplest approach. Call `get_filter_context(query)` per query and prepend it to the agent prompt. The agent sees available fields, stored values, extracted filters, and instructions — and decides what filters to pass to the retrieval tool:

```python
def search_documents_with_context(query: str) -> str:
    """Search the document knowledge base for relevant information."""
    # Give the agent full filter context — it decides what to apply
    context = rag.get_filter_context(query)
    filters = rag.extract_filters(query)  # agent can override this

    results = rag.retrieve(query, top_k=5, filters=filters)
    if not results:
        return "No relevant documents found."

    chunks = [context]  # prepend filter context so agent can reason about results
    for doc in results:
        source = doc.metadata.get("file_name", "unknown")
        chunks.append(f"[{source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(chunks)
```

### Option B — Static system prompt from stored values

Build a metadata-aware system prompt once at startup using `filter_fields` and `get_field_values()`:

```python
# filter_fields returns only semantic/filterable fields — excludes system fields
# like file_hash, chunk_id, source that are not useful for filtering
values = rag.get_field_values(rag.filter_fields)
# → {
#     'company_name': ['apple', 'microsoft', 'google'],
#     'doc_type':     ['10-k', '10-q'],
#     'fiscal_year':  [2024, 2025],
# }

SYSTEM_PROMPT = f"""
You are a helpful financial document assistant.
Use the search_documents tool to retrieve relevant information before answering.
Always cite the source document in your answer.

The knowledge base contains documents with the following metadata:
- company_name: {values['company_name']}
- doc_type: {values['doc_type']}
- fiscal_year: {values['fiscal_year']}

When calling search_documents, pass explicit filters based on what the user mentions.
"""
```

!!! note "auto_filter is off by default"
    Filters are not automatically extracted from queries unless you set `auto_filter: true` in `config.yaml`. For agents, keep it `false` and use `extract_filters()` or `get_filter_context()` to control filter extraction explicitly.

---

## 4. Create the Agent

Pass the metadata-aware `SYSTEM_PROMPT` from the previous step so the agent knows what's in the knowledge base.

=== "OpenAI"

    ```python
    from langchain.agents import create_agent
    from langchain_openai import ChatOpenAI

    model = ChatOpenAI(model="gpt-5.4-nano")

    agent = create_agent(
        model=model,
        tools=[search_documents],
        system_prompt=SYSTEM_PROMPT,
    )
    ```

=== "Ollama (local)"

    ```python
    from langchain.agents import create_agent
    from langchain_ollama import ChatOllama

    model = ChatOllama(model="qwen3.5:9b")

    agent = create_agent(
        model=model,
        tools=[search_documents],
        system_prompt=SYSTEM_PROMPT,
    )
    ```

=== "Groq"

    ```python
    from langchain.agents import create_agent
    from langchain_groq import ChatGroq

    model = ChatGroq(model="qwen/qwen3-32b")

    agent = create_agent(
        model=model,
        tools=[search_documents],
        system_prompt=SYSTEM_PROMPT,
    )
    ```

=== "Anthropic"

    ```python
    from langchain.agents import create_agent
    from langchain_anthropic import ChatAnthropic

    model = ChatAnthropic(model="claude-sonnet-4-6")

    agent = create_agent(
        model=model,
        tools=[search_documents],
        system_prompt=SYSTEM_PROMPT,
    )
    ```

---

## 5. Ask a Question

```python
response = agent.invoke({
    "messages": [{"role": "user", "content": "What is Apple's total revenue for 2025?"}]
})
print(response["messages"][-1].content)
```

---

## 6. Multi-Turn Conversation (Memory)

Add `InMemorySaver` to give the agent memory across turns within a session:

```python
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

model = ChatOpenAI(model="gpt-5.4-nano")
checkpointer = InMemorySaver()

agent = create_agent(
    model=model,
    tools=[search_documents],
    system_prompt=(
        "You are a helpful document assistant. "
        "Use the search_documents tool to answer questions."
    ),
    checkpointer=checkpointer,
)

config = {"configurable": {"thread_id": "session-1"}}

# Turn 1
response = agent.invoke(
    {"messages": [{"role": "user", "content": "What is Apple's revenue?"}]},
    config=config,
)
print(response["messages"][-1].content)

# Turn 2 — agent remembers the previous question
response = agent.invoke(
    {"messages": [{"role": "user", "content": "How does that compare to Microsoft?"}]},
    config=config,
)
print(response["messages"][-1].content)
```

!!! tip "Thread IDs"
    Each unique `thread_id` is a separate conversation. Reuse the same `thread_id` to continue a session; use a new one to start fresh.

---

## 7. Structured Output

Get a typed response with answer, sources, and confidence:

```python
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_openai import ChatOpenAI

@dataclass
class RAGResponse:
    """Structured agent response."""
    answer: str
    sources: list[str]
    confidence: str  # "high" | "medium" | "low"

model = ChatOpenAI(model="gpt-5.4-nano")

agent = create_agent(
    model=model,
    tools=[search_documents],
    system_prompt="You are a helpful document assistant. Use the search_documents tool to answer questions.",
    response_format=ToolStrategy(RAGResponse),
)

response = agent.invoke({
    "messages": [{"role": "user", "content": "What is Apple's net income for 2025?"}]
})
result = response["structured_response"]
print(f"Answer:     {result.answer}")
print(f"Sources:    {', '.join(result.sources)}")
print(f"Confidence: {result.confidence}")
```

---

## 8. Full Working Example

Save as `examples/rag_agent.py` and run:

```bash
python examples/rag_agent.py
```

```python
"""
RAG Agent — RAGWire + LangChain create_agent
============================================
Prerequisites:
  pip install "ragwire[openai]" fastembed langgraph
  export OPENAI_API_KEY="sk-..."

Place PDF files in examples/data/ then run:
  python examples/rag_agent.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from ragwire import RAGWire, setup_logging

logger = setup_logging(log_level="INFO")

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
DATA_DIR = Path(__file__).parent / "data"


# ------------------------------------------------------------------ #
# 1. Pipeline
# ------------------------------------------------------------------ #
rag = RAGWire(str(CONFIG_PATH))
stats = rag.ingest_directory(str(DATA_DIR))
logger.info(f"Ingested {stats['processed']} docs, {stats['chunks_created']} chunks")


# ------------------------------------------------------------------ #
# 2. Retrieval tool
# ------------------------------------------------------------------ #
@tool
def search_documents(query: str) -> str:
    """Search the document knowledge base for relevant information.

    Use this whenever the user asks a question that may be answered
    by the ingested documents.
    """
    results = rag.retrieve(query, top_k=5)
    if not results:
        return "No relevant documents found."

    chunks = []
    for doc in results:
        source = doc.metadata.get("file_name", "unknown")
        company = doc.metadata.get("company_name", "")
        year = doc.metadata.get("fiscal_year", "")
        header = f"[{source}" + (f" | {company} {year}" if company else "") + "]"
        chunks.append(f"{header}\n{doc.page_content}")

    return "\n\n---\n\n".join(chunks)


# ------------------------------------------------------------------ #
# 3. Agent with memory
# ------------------------------------------------------------------ #
model = ChatOpenAI(model="gpt-5.4-nano")
checkpointer = InMemorySaver()

agent = create_agent(
    model=model,
    tools=[search_documents],
    system_prompt=(
        "You are a helpful financial document assistant. "
        "Use the search_documents tool to retrieve relevant information "
        "from the knowledge base before answering questions. "
        "Always cite the source document in your answer."
    ),
    checkpointer=checkpointer,
)

config = {"configurable": {"thread_id": "demo"}}


# ------------------------------------------------------------------ #
# 4. Interactive Q&A loop
# ------------------------------------------------------------------ #
print("\nRAG Agent ready. Type 'quit' to exit.\n")

while True:
    question = input("You: ").strip()
    if question.lower() in ("quit", "exit", "q"):
        break
    if not question:
        continue

    response = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config=config,
    )
    print(f"\nAgent: {response['messages'][-1].content}\n")
```
