# Video 05 — Stateful RAG Pipeline with LangGraph

**Framework**: LangGraph
**Application**: Typed stateful graph that orchestrates ingestion → retrieval → generation with routing logic
**Difficulty**: Intermediate

---

## What You'll Build

A LangGraph graph with explicit nodes and typed state:
- `route` node — classifies query as "factual", "comparative", or "summary"
- `retrieve` node — runs RAGWire hybrid search with auto-filters
- `grade_docs` node — filters out irrelevant retrieved chunks
- `generate` node — produces final answer
- `fallback` node — handles no-result cases

---

## Install

```bash
pip install ragwire langgraph langchain-ollama
```

---

## Code: `pipeline.py`

```python
from typing import TypedDict, Annotated, List, Literal
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from ragwire import RAGWire, Config
import operator

# --- State Definition ---

class RAGState(TypedDict):
    query: str
    query_type: Literal["factual", "comparative", "summary"]
    filters: dict
    documents: Annotated[List[Document], operator.add]
    graded_documents: List[Document]
    answer: str
    confidence: Literal["high", "medium", "low"]

# --- Initialize ---

config = Config("config.yaml")
pipeline = RAGWire(config)
llm = ChatOllama(model="qwen2.5:7b")

# --- Nodes ---

def route_query(state: RAGState) -> RAGState:
    """Classify the query type to choose retrieval strategy."""
    prompt = f"""Classify this query into exactly one type:
- factual: asks for a specific fact or number
- comparative: compares multiple entities
- summary: asks for an overview or summary

Query: {state['query']}
Type (respond with only the word):"""

    result = llm.invoke([HumanMessage(content=prompt)])
    query_type = result.content.strip().lower()
    if query_type not in ("factual", "comparative", "summary"):
        query_type = "factual"

    return {**state, "query_type": query_type}


def extract_filters(state: RAGState) -> RAGState:
    """Use RAGWire auto-filter to extract metadata filters from query."""
    filters = pipeline.extract_filters(state["query"])
    return {**state, "filters": filters or {}}


def retrieve_documents(state: RAGState) -> RAGState:
    """Run RAGWire retrieval with extracted filters."""
    top_k = 8 if state["query_type"] == "comparative" else 5

    docs = pipeline.retrieve(
        state["query"],
        filters=state["filters"] if state["filters"] else None,
        top_k=top_k
    )
    return {**state, "documents": docs}


def grade_documents(state: RAGState) -> RAGState:
    """Filter docs to only those relevant to the query."""
    graded = []
    for doc in state["documents"]:
        prompt = f"""Is this document chunk relevant to the query?
Query: {state['query']}
Chunk: {doc.page_content[:400]}
Answer yes or no:"""
        result = llm.invoke([HumanMessage(content=prompt)])
        if "yes" in result.content.lower():
            graded.append(doc)

    return {**state, "graded_documents": graded}


def generate_answer(state: RAGState) -> RAGState:
    """Generate final answer from graded documents."""
    context = "\n\n---\n\n".join(d.page_content for d in state["graded_documents"])

    type_instructions = {
        "factual": "Provide a precise, concise answer with specific numbers/facts.",
        "comparative": "Create a structured comparison table or bullet list.",
        "summary": "Write a comprehensive summary with key themes."
    }

    messages = [
        SystemMessage(content=f"You are a document analyst. {type_instructions[state['query_type']]}"),
        HumanMessage(content=f"Context:\n{context}\n\nQuery: {state['query']}")
    ]

    result = llm.invoke(messages)
    confidence = "high" if len(state["graded_documents"]) >= 3 else "medium"

    return {**state, "answer": result.content, "confidence": confidence}


def fallback_answer(state: RAGState) -> RAGState:
    """Return honest fallback when no relevant documents found."""
    return {
        **state,
        "answer": ("I couldn't find relevant information to answer your question. "
                   "The documents may not contain this information, or try rephrasing."),
        "confidence": "low"
    }

# --- Routing Functions ---

def should_fallback(state: RAGState) -> Literal["generate", "fallback"]:
    return "fallback" if not state["graded_documents"] else "generate"

# --- Build Graph ---

def build_rag_graph() -> StateGraph:
    graph = StateGraph(RAGState)

    graph.add_node("route_query", route_query)
    graph.add_node("extract_filters", extract_filters)
    graph.add_node("retrieve", retrieve_documents)
    graph.add_node("grade_docs", grade_documents)
    graph.add_node("generate", generate_answer)
    graph.add_node("fallback", fallback_answer)

    graph.set_entry_point("route_query")
    graph.add_edge("route_query", "extract_filters")
    graph.add_edge("extract_filters", "retrieve")
    graph.add_edge("retrieve", "grade_docs")
    graph.add_conditional_edges("grade_docs", should_fallback)
    graph.add_edge("generate", END)
    graph.add_edge("fallback", END)

    return graph.compile()


rag_graph = build_rag_graph()


def ask(query: str) -> dict:
    initial_state = RAGState(
        query=query,
        query_type="factual",
        filters={},
        documents=[],
        graded_documents=[],
        answer="",
        confidence="low"
    )
    result = rag_graph.invoke(initial_state)
    print(f"\nQuery type: {result['query_type']}")
    print(f"Filters applied: {result['filters']}")
    print(f"Docs retrieved: {len(result['documents'])} → graded: {len(result['graded_documents'])}")
    print(f"Confidence: {result['confidence']}")
    print(f"\nAnswer:\n{result['answer']}")
    return result


if __name__ == "__main__":
    ask("What was Apple's net income in fiscal year 2024?")
    ask("Compare Apple and Microsoft's revenue growth")
    ask("Summarize the risk factors in the latest 10-K")
```

---

## Key RAGWire Concepts Covered

| Concept | Code location |
|---------|---------------|
| `pipeline.extract_filters()` | LLM-based metadata filter extraction |
| `pipeline.retrieve(top_k=...)` | Configurable retrieval depth |
| Typed `RAGState` | Structured pipeline state |
| Document grading | LLM relevance scoring loop |

---

## What to Explain in Video

1. Why LangGraph over plain Python — state visibility, debugging (5 min)
2. `TypedDict` state design — what goes in state (5 min)
3. Routing node pattern — classify before retrieve (5 min)
4. Document grading — why not all retrieved docs are useful (5 min)
5. Conditional edges — `should_fallback` routing (5 min)
6. Drawing the graph with `graph.get_graph().draw_mermaid()` (3 min)
7. Live demo with trace output (7 min)
