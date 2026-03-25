# Video 09 — Autonomous Document Intelligence Agent with DeepAgent

**Framework**: DeepAgent (deep-agent / AutoGen-style autonomous loop)
**Application**: Fully autonomous agent that ingests, analyzes, and produces structured reports without human input
**Difficulty**: Intermediate–Advanced

---

## What You'll Build

A fully autonomous document intelligence agent using a ReAct-style loop:
- Autonomously ingests a document directory
- Plans its own analysis strategy
- Calls RAGWire tools repeatedly to gather evidence
- Produces a structured intelligence report
- Self-terminates when confidence is sufficient

Uses the `deep-agent` pattern: an LLM agent in an `observe → think → act` loop.

---

## Install

```bash
pip install ragwire langchain langchain-ollama langgraph
```

> Note: "DeepAgent" in this tutorial refers to the deep reasoning autonomous agent pattern
> (observe → think → act loop with self-termination), not a specific package.
> We implement it with LangGraph + ReAct.

---

## Code: `doc_intelligence_agent.py`

```python
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from ragwire import RAGWire, Config
import json
import os

config = Config("config.yaml")
pipeline = RAGWire(config)
llm = ChatOllama(model="qwen2.5:7b")

# --- Tools ---

@tool
def ingest_documents(directory_path: str) -> str:
    """
    Ingest all documents from a directory into the RAGWire pipeline.
    Returns ingestion statistics including how many chunks were created.
    """
    if not os.path.exists(directory_path):
        return f"Directory not found: {directory_path}"
    stats = pipeline.ingest_directory(directory_path, recursive=True)
    return (f"Ingestion complete: {stats['processed']} files processed, "
            f"{stats['chunks_created']} chunks created, "
            f"{stats['skipped']} skipped (duplicates), "
            f"{stats['failed']} failed.")

@tool
def explore_collection() -> str:
    """
    Discover what metadata fields and values exist in the document collection.
    Call this before searching to understand what data is available.
    """
    fields = pipeline.discover_metadata_fields()
    values = pipeline.get_field_values(list(fields.keys())[:5])
    return json.dumps({"fields": fields, "sample_values": values}, indent=2)

@tool
def search(query: str, filters_json: str = "{}") -> str:
    """
    Search documents with hybrid retrieval.
    filters_json: JSON string of metadata filters, e.g. '{"company_name": ["apple inc"]}'
    Returns top matching document passages.
    """
    try:
        filters = json.loads(filters_json)
    except Exception:
        filters = {}

    docs = pipeline.retrieve(query, filters=filters if filters else None, top_k=5)
    if not docs:
        return "No results found."

    results = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        results.append(
            f"[{i}] Source: {meta.get('file_name', 'unknown')} | "
            f"{meta.get('company_name', '')} {meta.get('fiscal_year', '')}\n"
            f"{doc.page_content[:700]}"
        )
    return "\n\n---\n\n".join(results)

@tool
def extract_structured_data(query: str, fields: str) -> str:
    """
    Extract specific structured data fields by searching documents.
    fields: comma-separated list of fields to extract, e.g. "revenue, net_income, debt"
    Returns a structured extraction attempt.
    """
    field_list = [f.strip() for f in fields.split(",")]
    results = {}
    for field in field_list:
        docs = pipeline.retrieve(f"{field} financial data numbers", top_k=3)
        if docs:
            results[field] = docs[0].page_content[:400]
        else:
            results[field] = "Not found"
    return json.dumps(results, indent=2)

@tool
def write_report(title: str, content: str, filename: str = "intelligence_report.md") -> str:
    """
    Save the final intelligence report to a markdown file.
    Call this as the FINAL action when analysis is complete.
    """
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n{content}")
    return f"Report saved to {filename}"

# --- Build Agent ---

SYSTEM_PROMPT = """You are an autonomous document intelligence agent.
Your mission is to analyze documents and produce a comprehensive intelligence report.

Workflow:
1. First, ingest documents if needed (use ingest_documents)
2. Explore the collection to understand what's available (use explore_collection)
3. Plan your analysis: what questions to answer, what metrics to extract
4. Execute multiple targeted searches to gather evidence
5. Extract structured data for key metrics
6. When confident, write the final report (use write_report)
7. After writing the report, STOP — your job is done.

Be systematic, thorough, and cite evidence. Self-terminate after writing the report."""

memory = InMemorySaver()
agent = create_agent(
    llm,
    tools=[ingest_documents, explore_collection, search, extract_structured_data, write_report],
    checkpointer=memory,
    system_prompt=SYSTEM_PROMPT
)

def run_autonomous_analysis(
    directory: str,
    analysis_goal: str,
    thread_id: str = "intel-001"
) -> str:
    """
    Run a fully autonomous document analysis session.

    Args:
        directory: Path to documents to analyze
        analysis_goal: High-level goal for the analysis
        thread_id: Unique session identifier
    """
    config = {"configurable": {"thread_id": thread_id}}

    task = f"""
    Analyze the documents in: {directory}

    Analysis Goal: {analysis_goal}

    Produce a comprehensive intelligence report covering:
    - Key entities and their characteristics
    - Important metrics and data points
    - Trends and patterns
    - Key risks or concerns
    - Strategic implications

    Save the final report when complete.
    """

    print(f"Starting autonomous analysis of: {directory}")
    print(f"Goal: {analysis_goal}\n")
    print("="*60)

    response = agent.invoke(
        {"messages": [HumanMessage(content=task)]},
        config=config
    )

    # Print agent's reasoning trace
    for msg in response["messages"]:
        if isinstance(msg, AIMessage):
            if msg.content:
                print(f"\nAgent: {msg.content[:500]}")

    return response["messages"][-1].content


if __name__ == "__main__":
    result = run_autonomous_analysis(
        directory="examples/data",
        analysis_goal=(
            "Analyze Apple's financial performance, strategic position, "
            "and key risks from their SEC filings"
        )
    )
    print("\n" + "="*60)
    print("Autonomous analysis complete.")
    print("Check intelligence_report.md for the full report.")
```

---

## Key Concepts Covered

| Concept | Code location |
|---------|---------------|
| Autonomous tool selection | Agent plans own analysis sequence |
| Self-termination | `write_report` → agent stops |
| `ingest_documents` tool | Agent can ingest its own data |
| `explore_collection` first | Meta-reasoning before search |
| Multi-step evidence gathering | Multiple `search` calls |

---

## What to Explain in Video

1. What "autonomous agent" means — observe, think, act loop (5 min)
2. Tool design for autonomous agents — enabling vs constraining (7 min)
3. Self-termination — why `write_report` is the stop signal (5 min)
4. Tracing agent reasoning from message history (5 min)
5. When autonomous agents go wrong — common failure modes (5 min)
6. Live demo with trace output showing agent planning (13 min)
