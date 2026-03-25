# Video 12 — Interactive Workflow Builder with LangGraph + Streamlit

**Framework**: LangGraph + Streamlit
**Application**: Visual human-in-the-loop document workflow — user approves each step before the agent proceeds
**Difficulty**: Advanced

---

## What You'll Build

A Streamlit app with a human-in-the-loop LangGraph workflow:
- User selects a document analysis task
- Agent runs Step 1 (document discovery) → pauses for human approval
- User reviews, edits, and approves the plan
- Agent runs Step 2 (retrieval) → pauses, shows results → human filters
- Agent runs Step 3 (generation) → human approves or requests regeneration
- Final report saved and displayed

Uses LangGraph's `interrupt` mechanism for human-in-the-loop control.

---

## Install

```bash
pip install ragwire langgraph langchain-ollama streamlit
```

---

## Code: `workflow_app.py`

```python
import streamlit as st
from typing import TypedDict, Literal, List, Optional
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt
from ragwire import RAGWire, Config

st.set_page_config(page_title="RAG Workflow Builder", layout="wide")
st.title("Interactive RAG Workflow Builder")

@st.cache_resource
def load_pipeline():
    return RAGWire(Config("config.yaml"))

pipeline = load_pipeline()
llm = ChatOllama(model="qwen2.5:7b")

# --- LangGraph State ---

class WorkflowState(TypedDict):
    task: str
    analysis_plan: str
    search_queries: List[str]
    documents: List[Document]
    approved_docs: List[Document]
    final_report: str
    human_feedback: Optional[str]
    step: str

# --- Graph Nodes ---

def plan_analysis(state: WorkflowState) -> WorkflowState:
    """Generate an analysis plan for the task."""
    result = llm.invoke([
        HumanMessage(content=f"""Create a concise analysis plan for this task:
Task: {state['task']}

Output:
1. Analysis plan (3-5 bullets)
2. Search queries to use (3-5 queries as a JSON list)

Format:
PLAN:
- ...

QUERIES:
["query1", "query2", ...]""")
    ])

    content = result.content
    # Parse plan and queries
    plan = ""
    queries = []
    if "PLAN:" in content:
        plan = content.split("PLAN:")[1].split("QUERIES:")[0].strip()
    if "QUERIES:" in content:
        import json, re
        raw = content.split("QUERIES:")[1].strip()
        match = re.search(r'\[.*?\]', raw, re.DOTALL)
        if match:
            try:
                queries = json.loads(match.group())
            except Exception:
                queries = [state["task"]]

    return {**state, "analysis_plan": plan, "search_queries": queries, "step": "planned"}


def human_approve_plan(state: WorkflowState) -> WorkflowState:
    """Pause for human to review and approve the analysis plan."""
    feedback = interrupt({
        "type": "approve_plan",
        "plan": state["analysis_plan"],
        "queries": state["search_queries"],
        "message": "Review the analysis plan. Approve or provide feedback."
    })
    return {**state, "human_feedback": feedback, "step": "plan_approved"}


def execute_search(state: WorkflowState) -> WorkflowState:
    """Run all search queries and collect documents."""
    all_docs = []
    seen_ids = set()

    for query in state["search_queries"]:
        docs = pipeline.retrieve(query, top_k=3)
        for doc in docs:
            doc_id = hash(doc.page_content[:100])
            if doc_id not in seen_ids:
                all_docs.append(doc)
                seen_ids.add(doc_id)

    return {**state, "documents": all_docs, "step": "searched"}


def human_filter_docs(state: WorkflowState) -> WorkflowState:
    """Pause for human to review and select relevant documents."""
    feedback = interrupt({
        "type": "filter_docs",
        "documents": [
            {
                "index": i,
                "source": d.metadata.get("file_name", "unknown"),
                "preview": d.page_content[:200]
            }
            for i, d in enumerate(state["documents"])
        ],
        "message": "Select which documents to include (comma-separated indices, or 'all')."
    })

    if feedback == "all" or not feedback:
        approved = state["documents"]
    else:
        try:
            indices = [int(x.strip()) for x in feedback.split(",")]
            approved = [state["documents"][i] for i in indices if i < len(state["documents"])]
        except Exception:
            approved = state["documents"]

    return {**state, "approved_docs": approved, "step": "docs_filtered"}


def generate_report(state: WorkflowState) -> WorkflowState:
    """Generate the final report from approved documents."""
    context = "\n\n---\n\n".join(d.page_content for d in state["approved_docs"])

    result = llm.invoke([
        HumanMessage(content=f"""Complete this analysis task using the provided context.

Task: {state['task']}
Analysis Plan: {state['analysis_plan']}

Context:
{context}

Write a comprehensive, well-structured report.""")
    ])

    return {**state, "final_report": result.content, "step": "generated"}


def human_approve_report(state: WorkflowState) -> WorkflowState:
    """Final human review before saving."""
    feedback = interrupt({
        "type": "approve_report",
        "report": state["final_report"],
        "message": "Approve report or provide feedback for regeneration."
    })
    return {**state, "human_feedback": feedback, "step": "report_reviewed"}


def should_regenerate(state: WorkflowState) -> Literal["regenerate", "done"]:
    feedback = state.get("human_feedback", "").lower()
    return "regenerate" if feedback and feedback not in ("approve", "ok", "yes", "done", "") else "done"

# --- Build Graph ---

@st.cache_resource
def build_workflow():
    graph = StateGraph(WorkflowState)
    graph.add_node("plan", plan_analysis)
    graph.add_node("approve_plan", human_approve_plan)
    graph.add_node("search", execute_search)
    graph.add_node("filter_docs", human_filter_docs)
    graph.add_node("generate", generate_report)
    graph.add_node("approve_report", human_approve_report)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "approve_plan")
    graph.add_edge("approve_plan", "search")
    graph.add_edge("search", "filter_docs")
    graph.add_edge("filter_docs", "generate")
    graph.add_edge("generate", "approve_report")
    graph.add_conditional_edges(
        "approve_report",
        should_regenerate,
        {"regenerate": "generate", "done": END}
    )

    memory = InMemorySaver()
    return graph.compile(
        checkpointer=memory,
        interrupt_before=["approve_plan", "filter_docs", "approve_report"]
    )

workflow = build_workflow()

# --- Streamlit UI ---

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = "workflow-001"
if "workflow_state" not in st.session_state:
    st.session_state["workflow_state"] = None

thread_config = {"configurable": {"thread_id": st.session_state["thread_id"]}}

# Task input
task = st.text_area("Analysis Task", placeholder="e.g. Analyze Apple's revenue growth strategy and outlook")
if st.button("Start Workflow") and task:
    initial = WorkflowState(
        task=task, analysis_plan="", search_queries=[],
        documents=[], approved_docs=[], final_report="",
        human_feedback=None, step="start"
    )
    result = workflow.invoke(initial, config=thread_config)
    st.session_state["workflow_state"] = result
    st.rerun()

# Handle interrupt states
state = st.session_state.get("workflow_state")
if state:
    current_step = state.get("step", "")
    st.markdown(f"**Current step**: `{current_step}`")

    if current_step == "planned":
        st.subheader("Review Analysis Plan")
        st.markdown(state["analysis_plan"])
        st.markdown("**Search queries:**")
        for q in state["search_queries"]:
            st.markdown(f"- `{q}`")

        feedback = st.text_input("Feedback (or leave empty to approve)")
        if st.button("Approve Plan"):
            result = workflow.invoke(
                {"type": "approve_plan", "feedback": feedback},
                config=thread_config
            )
            st.session_state["workflow_state"] = result
            st.rerun()

    elif current_step == "searched":
        st.subheader("Select Documents to Include")
        selections = []
        for i, doc in enumerate(state["documents"]):
            include = st.checkbox(
                f"[{i}] {doc.metadata.get('file_name', 'doc')} — {doc.page_content[:100]}...",
                value=True, key=f"doc_{i}"
            )
            if include:
                selections.append(str(i))

        if st.button("Confirm Document Selection"):
            result = workflow.invoke(
                ",".join(selections) or "all",
                config=thread_config
            )
            st.session_state["workflow_state"] = result
            st.rerun()

    elif current_step == "generated":
        st.subheader("Review Final Report")
        st.markdown(state["final_report"])
        feedback = st.text_input("Feedback (leave empty to approve, or describe changes needed)")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Approve Report"):
                result = workflow.invoke("approve", config=thread_config)
                st.session_state["workflow_state"] = result
                st.rerun()
        with col2:
            if st.button("Regenerate") and feedback:
                result = workflow.invoke(feedback, config=thread_config)
                st.session_state["workflow_state"] = result
                st.rerun()

    elif state.get("final_report") and current_step not in ("planned", "searched", "generated"):
        st.success("Workflow complete!")
        st.markdown(state["final_report"])
        st.download_button("Download Report", state["final_report"], "report.md", "text/markdown")
```

---

## Key Concepts Covered

| Concept | Code location |
|---------|---------------|
| `interrupt()` in nodes | Pause graph for human input |
| `interrupt_before=[...]` | Declarative interrupt points |
| `workflow.invoke(feedback)` | Resume with human input |
| `InMemorySaver` + `thread_id` | Persistent graph state between runs |
| Conditional regeneration | `should_regenerate` edge function |

---

## What to Explain in Video

1. What is human-in-the-loop and why it matters for RAG (5 min)
2. LangGraph `interrupt()` mechanism — how it pauses execution (7 min)
3. `interrupt_before` vs `interrupt_after` (3 min)
4. Resuming a graph with `invoke(human_input)` (5 min)
5. Streamlit `st.rerun()` to update UI after graph steps (5 min)
6. Building review UIs for each interrupt type (5 min)
7. Live walkthrough of the full workflow (10 min)
