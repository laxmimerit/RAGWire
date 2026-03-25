# Video 06 — Self-Correcting RAG Agent with LangGraph

**Framework**: LangGraph
**Application**: Agentic RAG that grades its own answers, rewrites queries, and retries until confidence threshold is met
**Difficulty**: Intermediate–Advanced

---

## What You'll Build

A LangGraph agent with a **correction loop**:
1. Retrieve documents → Grade relevance
2. Generate answer → Grade answer quality
3. If answer is poor → Rewrite query → Retry (max 3 iterations)
4. If still poor → Escalate to web search fallback
5. Return final answer with iteration count and quality score

---

## Install

```bash
pip install ragwire langgraph langchain-ollama langchain-community
```

---

## Code: `self_correcting_rag.py`

```python
from typing import TypedDict, Literal, List
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from ragwire import RAGWire, Config
import json

config = Config("config.yaml")
pipeline = RAGWire(config)
llm = ChatOllama(model="qwen2.5:7b", format="json")
llm_text = ChatOllama(model="qwen2.5:7b")

MAX_ITERATIONS = 3

# --- State ---

class CorrectionState(TypedDict):
    original_query: str
    current_query: str
    iteration: int
    documents: List[Document]
    answer: str
    doc_relevance_score: float   # 0.0 to 1.0
    answer_quality_score: float  # 0.0 to 1.0
    rewrite_reason: str
    final: bool

# --- Nodes ---

def retrieve(state: CorrectionState) -> CorrectionState:
    docs = pipeline.retrieve(state["current_query"], top_k=5)
    return {**state, "documents": docs}


def grade_documents(state: CorrectionState) -> CorrectionState:
    """Score how relevant the retrieved docs are to the query (0-1)."""
    if not state["documents"]:
        return {**state, "doc_relevance_score": 0.0}

    scores = []
    for doc in state["documents"]:
        prompt = f"""Rate relevance of this chunk to the query from 0.0 to 1.0.
Query: {state['current_query']}
Chunk: {doc.page_content[:300]}
Respond with JSON: {{"score": 0.7}}"""
        try:
            result = llm.invoke([HumanMessage(content=prompt)])
            score = json.loads(result.content).get("score", 0.5)
            scores.append(float(score))
        except Exception:
            scores.append(0.5)

    avg_score = sum(scores) / len(scores)
    return {**state, "doc_relevance_score": avg_score}


def generate(state: CorrectionState) -> CorrectionState:
    if not state["documents"]:
        return {**state, "answer": "No relevant documents found."}

    context = "\n\n---\n\n".join(d.page_content for d in state["documents"])
    messages = [
        SystemMessage(content="Answer precisely using only the provided context."),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {state['original_query']}")
    ]
    result = llm_text.invoke(messages)
    return {**state, "answer": result.content}


def grade_answer(state: CorrectionState) -> CorrectionState:
    """Score answer quality — does it actually answer the original question?"""
    prompt = f"""Evaluate if this answer adequately addresses the question.
Question: {state['original_query']}
Answer: {state['answer']}

Consider: completeness, specificity, factual grounding.
Respond with JSON: {{"score": 0.8, "reason": "missing specific numbers"}}"""

    try:
        result = llm.invoke([HumanMessage(content=prompt)])
        data = json.loads(result.content)
        score = float(data.get("score", 0.5))
        reason = data.get("reason", "")
    except Exception:
        score, reason = 0.5, "grading failed"

    return {**state, "answer_quality_score": score, "rewrite_reason": reason}


def rewrite_query(state: CorrectionState) -> CorrectionState:
    """Rewrite the query to improve retrieval based on why the answer failed."""
    prompt = f"""The current query failed to retrieve useful information.
Original query: {state['original_query']}
Current query: {state['current_query']}
Reason answer was poor: {state['rewrite_reason']}

Write a better, more specific search query. Respond with just the query text."""

    result = llm_text.invoke([HumanMessage(content=prompt)])
    new_query = result.content.strip()

    return {
        **state,
        "current_query": new_query,
        "iteration": state["iteration"] + 1,
        "documents": [],
        "answer": ""
    }


def web_search_fallback(state: CorrectionState) -> CorrectionState:
    """Fallback when RAG fails after max iterations."""
    # In a real app, integrate with Tavily/DuckDuckGo here
    return {
        **state,
        "answer": (f"After {state['iteration']} retrieval attempts, "
                   f"the documents don't contain sufficient information. "
                   f"Consider searching the web or consulting primary sources."),
        "final": True
    }

# --- Routing ---

def should_retry(state: CorrectionState) -> Literal["rewrite", "done", "web_fallback"]:
    quality = state.get("answer_quality_score", 0)
    relevance = state.get("doc_relevance_score", 0)
    iterations = state.get("iteration", 0)

    if quality >= 0.7 and relevance >= 0.6:
        return "done"
    elif iterations >= MAX_ITERATIONS:
        return "web_fallback"
    else:
        return "rewrite"

# --- Build Graph ---

def build_graph():
    graph = StateGraph(CorrectionState)

    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_docs", grade_documents)
    graph.add_node("generate", generate)
    graph.add_node("grade_answer", grade_answer)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("web_fallback", web_search_fallback)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade_docs")
    graph.add_edge("grade_docs", "generate")
    graph.add_edge("generate", "grade_answer")
    graph.add_conditional_edges(
        "grade_answer",
        should_retry,
        {"rewrite": "rewrite_query", "done": END, "web_fallback": "web_fallback"}
    )
    graph.add_edge("rewrite_query", "retrieve")  # Loop back!
    graph.add_edge("web_fallback", END)

    return graph.compile()


rag_graph = build_graph()


def ask(query: str) -> dict:
    initial = CorrectionState(
        original_query=query,
        current_query=query,
        iteration=0,
        documents=[],
        answer="",
        doc_relevance_score=0.0,
        answer_quality_score=0.0,
        rewrite_reason="",
        final=False
    )
    result = rag_graph.invoke(initial)

    print(f"Iterations: {result['iteration']}")
    print(f"Doc relevance: {result['doc_relevance_score']:.2f}")
    print(f"Answer quality: {result['answer_quality_score']:.2f}")
    print(f"Final query used: {result['current_query']}")
    print(f"\nAnswer:\n{result['answer']}")
    return result


if __name__ == "__main__":
    ask("What specific revenue guidance did Apple provide for Q2 2025?")
```

---

## Key Concepts Covered

| Concept | Code location |
|---------|---------------|
| Correction loop | `rewrite_query → retrieve` edge cycle |
| LLM-as-judge pattern | `grade_documents`, `grade_answer` nodes |
| JSON-mode scoring | `ChatOllama(format="json")` |
| Max iteration guard | `iterations >= MAX_ITERATIONS` check |
| Conditional multi-branch | `should_retry` → 3 targets |

---

## What to Explain in Video

1. Why naive RAG fails — the relevance gap problem (5 min)
2. LLM-as-judge pattern — grading docs and answers (7 min)
3. Query rewriting — how and why it improves retrieval (5 min)
4. Loop detection in LangGraph — `MAX_ITERATIONS` guard (3 min)
5. JSON mode for structured LLM output (3 min)
6. Visualizing the correction loop with Mermaid (3 min)
7. Live demo showing iteration trace (9 min)
