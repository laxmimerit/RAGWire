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

## 2. Define Two Tools

The agent gets two tools with clear separation of concerns:

- **`get_filter_context`** — call when metadata awareness is needed. Returns available fields, stored values, extracted filter suggestions, and instructions. Always fresh from Qdrant — safe to call multiple times in multi-query flows.
- **`search_documents`** — pure retrieval. Accepts explicit filters the agent decided from the context.

```python
from langchain.tools import tool
from typing import Optional

@tool
def get_filter_context(query: str) -> str:
    """Get available metadata fields, stored values, and filter suggestions for a query.

    Call this before search_documents when the query may involve specific metadata
    (e.g. a company, year, document type, author). The returned context shows what
    filters are available and what was extracted from your query — use it to decide
    what filters to pass to search_documents.

    Skip this tool for purely semantic queries with no metadata intent.
    """
    return rag.get_filter_context(query)


@tool
def search_documents(query: str, filters: Optional[dict] = None) -> str:
    """Search the document knowledge base for relevant information.

    Args:
        query: The search query
        filters: Optional metadata filters decided from get_filter_context.
                 Pass {} or omit to search without filtering.
    """
    results = rag.retrieve(query, top_k=5, filters=filters)
    if not results:
        return "No relevant documents found."

    chunks = []
    for doc in results:
        source = doc.metadata.get("file_name", "unknown")
        meta_parts = [
            f"{k}={str(v)[:100]}"
            for k, v in doc.metadata.items()
            if k != "file_name" and v not in (None, "", [])
        ]
        header = f"[{source}" + (f" | {', '.join(meta_parts)}" if meta_parts else "") + "]"
        chunks.append(f"{header}\n{doc.page_content}")

    return "\n\n---\n\n".join(chunks)
```

**Agent reasoning flow:**

```
User: "creatine studies from 2023"

1. Agent calls get_filter_context("creatine studies from 2023")
   → sees research_focus values, publication_year: [2022, 2023, 2024]
   → extracted: research_focus: "creatine", publication_year: 2023
   → decides: {"publication_year": 2023}  (research_focus exact match uncertain)

2. Agent calls search_documents("creatine studies from 2023", filters={"publication_year": 2023})
   → clean retrieval with agent-decided filters
```

For multi-query tasks, the agent calls `get_filter_context` independently per sub-query — always fresh from Qdrant.

---

## 3. Create the Agent

Register both tools and give the agent a minimal system prompt:

=== "Ollama (local)"

    ```python
    from langchain.agents import create_agent
    from langchain_ollama import ChatOllama

    model = ChatOllama(model="qwen3.5:9b", base_url="http://localhost:11434")

    agent = create_agent(
        model=model,
        tools=[get_filter_context, search_documents],
        system_prompt=(
            "You are a helpful document assistant. "
            "For complex questions, break them down into simple sub-questions and answer each one before forming a final answer. "
            "Always use search_documents to retrieve information before answering — never answer from general knowledge. "
            "Use get_filter_context before search_documents when the query involves specific metadata (company, year, document type, etc.). "
            "If no relevant documents are found, say so — do not guess or fabricate an answer. "
            "Always cite the source document in your answer."
        ),
    )
    ```

=== "OpenAI"

    ```python
    from langchain.agents import create_agent
    from langchain_openai import ChatOpenAI

    model = ChatOpenAI(model="gpt-5.4-nano")

    agent = create_agent(
        model=model,
        tools=[get_filter_context, search_documents],
        system_prompt=(
            "You are a helpful document assistant. "
            "For complex questions, break them down into simple sub-questions and answer each one before forming a final answer. "
            "Always use search_documents to retrieve information before answering — never answer from general knowledge. "
            "Use get_filter_context before search_documents when the query involves specific metadata (company, year, document type, etc.). "
            "If no relevant documents are found, say so — do not guess or fabricate an answer. "
            "Always cite the source document in your answer."
        ),
    )
    ```

=== "Groq"

    ```python
    from langchain.agents import create_agent
    from langchain_groq import ChatGroq

    model = ChatGroq(model="qwen/qwen3-32b")

    agent = create_agent(
        model=model,
        tools=[get_filter_context, search_documents],
        system_prompt=(
            "You are a helpful document assistant. "
            "For complex questions, break them down into simple sub-questions and answer each one before forming a final answer. "
            "Always use search_documents to retrieve information before answering — never answer from general knowledge. "
            "Use get_filter_context before search_documents when the query involves specific metadata (company, year, document type, etc.). "
            "If no relevant documents are found, say so — do not guess or fabricate an answer. "
            "Always cite the source document in your answer."
        ),
    )
    ```

=== "Anthropic"

    ```python
    from langchain.agents import create_agent
    from langchain_anthropic import ChatAnthropic

    model = ChatAnthropic(model="claude-sonnet-4-6")

    agent = create_agent(
        model=model,
        tools=[get_filter_context, search_documents],
        system_prompt=(
            "You are a helpful document assistant. "
            "For complex questions, break them down into simple sub-questions and answer each one before forming a final answer. "
            "Always use search_documents to retrieve information before answering — never answer from general knowledge. "
            "Use get_filter_context before search_documents when the query involves specific metadata (company, year, document type, etc.). "
            "If no relevant documents are found, say so — do not guess or fabricate an answer. "
            "Always cite the source document in your answer."
        ),
    )
    ```

---

## 4. Ask a Question

```python
from langchain_core.messages import HumanMessage

response = agent.invoke({
    "messages": [HumanMessage("What is Apple's total revenue for 2025?")]
})
print(response["messages"][-1].content)
```

---

## 5. Multi-Turn Conversation (Memory)

Add `InMemorySaver` to give the agent memory across turns within a session:

```python
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver

model = ChatOllama(model="qwen3.5:9b", base_url="http://localhost:11434")
checkpointer = InMemorySaver()

agent = create_agent(
    model=model,
    tools=[search_documents],
    system_prompt=(
        "You are a helpful document assistant. "
        "For complex questions, break them down into simple sub-questions and answer each one before forming a final answer. "
        "Always use search_documents to retrieve information before answering — never answer from general knowledge. "
        "If no relevant documents are found, say so — do not guess or fabricate an answer. "
        "Always cite the source document in your answer."
    ),
    checkpointer=checkpointer,
)

config = {"configurable": {"thread_id": "session-1"}}

# Turn 1
response = agent.invoke(
    {"messages": [HumanMessage("What is Apple's revenue?")]},
    config=config,
)
print(response["messages"][-1].content)

# Turn 2 — agent remembers the previous question
response = agent.invoke(
    {"messages": [HumanMessage("How does that compare to Microsoft?")]},
    config=config,
)
print(response["messages"][-1].content)
```

!!! tip "Thread IDs"
    Each unique `thread_id` is a separate conversation. Reuse the same `thread_id` to continue a session; use a new one to start fresh.

---

## 6. Structured Output

Get a typed response with answer, sources, and confidence:

```python
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

@dataclass
class RAGResponse:
    """Structured agent response."""
    answer: str
    sources: list[str]
    confidence: str  # "high" | "medium" | "low"

model = ChatOllama(model="qwen3.5:9b", base_url="http://localhost:11434")

agent = create_agent(
    model=model,
    tools=[search_documents],
    system_prompt=(
        "You are a helpful document assistant. "
        "For complex questions, break them down into simple sub-questions and answer each one before forming a final answer. "
        "Always use search_documents to retrieve information before answering — never answer from general knowledge. "
        "If no relevant documents are found, say so — do not guess or fabricate an answer. "
        "Always cite the source document in your answer."
    ),
    response_format=ToolStrategy(RAGResponse),
)

response = agent.invoke({
    "messages": [HumanMessage("What is Apple's net income for 2025?")]
})
result = response["structured_response"]
print(f"Answer:     {result.answer}")
print(f"Sources:    {', '.join(result.sources)}")
print(f"Confidence: {result.confidence}")
```

---

## 7. Full Working Example

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

from typing import Optional

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver

from ragwire import RAGWire, setup_logging

logger = setup_logging(log_level="INFO")


# ------------------------------------------------------------------ #
# 1. Pipeline
# ------------------------------------------------------------------ #
rag = RAGWire("config.yaml")
stats = rag.ingest_directory("data/")
logger.info(f"Ingested {stats['processed']} docs, {stats['chunks_created']} chunks")


# ------------------------------------------------------------------ #
# 2. Tools
# ------------------------------------------------------------------ #
@tool
def get_filter_context(query: str) -> str:
    """Get available metadata fields, stored values, and filter suggestions for a query.

    Call this before search_documents when the query involves specific metadata
    (company, year, document type, etc.). Use the returned context to decide
    what filters to pass to search_documents.

    Skip this for purely semantic queries with no metadata intent.
    """
    return rag.get_filter_context(query)


@tool
def search_documents(query: str, filters: Optional[dict] = None) -> str:
    """Search the document knowledge base for relevant information.

    Args:
        query: The search query
        filters: Optional metadata filters decided from get_filter_context.
                 Pass {} or omit to search without filtering.
    """
    results = rag.retrieve(query, top_k=5, filters=filters)
    if not results:
        return "No relevant documents found."

    chunks = []
    for doc in results:
        source = doc.metadata.get("file_name", "unknown")
        meta_parts = [
            f"{k}={str(v)[:100]}"
            for k, v in doc.metadata.items()
            if k != "file_name" and v not in (None, "", [])
        ]
        header = f"[{source}" + (f" | {', '.join(meta_parts)}" if meta_parts else "") + "]"
        chunks.append(f"{header}\n{doc.page_content}")

    return "\n\n---\n\n".join(chunks)


# ------------------------------------------------------------------ #
# 3. Agent with memory
# ------------------------------------------------------------------ #
model = ChatOllama(model="qwen3.5:9b", base_url="http://localhost:11434")
checkpointer = InMemorySaver()

agent = create_agent(
    model=model,
    tools=[get_filter_context, search_documents],
    system_prompt=(
        "You are a helpful financial document assistant. "
        "For complex questions, break them down into simple sub-questions and answer each one before forming a final answer. "
        "Always use search_documents to retrieve information before answering — never answer from general knowledge. "
        "Use get_filter_context before search_documents when the query involves specific metadata (company, year, document type, etc.). "
        "If no relevant documents are found, say so — do not guess or fabricate an answer. "
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
        {"messages": [HumanMessage(question)]},
        config=config,
    )
    print(f"\nAgent: {response['messages'][-1].content}\n")
```
