# Video 04 — SEC Filing Analyzer Agent with LangChain

**Framework**: LangChain (tool-calling agent)
**Application**: Autonomous agent with RAGWire tools to analyze SEC filings and answer complex financial queries
**Difficulty**: Intermediate

---

## What You'll Build

A LangChain tool-calling agent that:
- Has access to two RAGWire tools: `get_filter_context` and `search_documents`
- Autonomously decides which company/year to filter for
- Answers multi-part financial questions (revenue, risk factors, guidance)
- Maintains conversation context across questions

---

## Install

```bash
pip install ragwire langchain langchain-ollama langgraph
```

---

## Project Structure

```
sec_agent/
├── agent.py
├── config.yaml
└── finance_metadata.yaml
```

---

## Code: `agent.py`

```python
from ragwire import RAGWire, Config
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

# Initialize pipeline
config = Config("config.yaml")
pipeline = RAGWire(config)
llm = ChatOllama(model="qwen2.5:7b")

# Tool 1 — understand what metadata is available
@tool
def get_filter_context(query: str) -> str:
    """
    Get available metadata fields and suggested filters for a query.
    Call this FIRST before searching to understand what companies/years are available.
    Returns a structured context block with field options and suggested filters.
    """
    return pipeline.get_filter_context(query)

# Tool 2 — search documents with optional filters
@tool
def search_documents(query: str, filters: dict = None) -> str:
    """
    Search documents using semantic + keyword hybrid retrieval.
    Optionally pass filters dict like {"company_name": ["apple inc"], "fiscal_year": [2024]}.
    Returns relevant document chunks with metadata.
    """
    docs = pipeline.retrieve(query, filters=filters)
    if not docs:
        return "No relevant documents found."

    results = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        header = (f"[{i}] {meta.get('company_name', 'Unknown')} | "
                  f"{meta.get('doc_type', '')} | FY{meta.get('fiscal_year', '')}")
        results.append(f"{header}\n{doc.page_content[:800]}")

    return "\n\n---\n\n".join(results)

# Tool 3 — compare across companies
@tool
def compare_companies(query: str, companies: list[str]) -> str:
    """
    Retrieve information about the same topic across multiple companies for comparison.
    Pass a list of company names to compare side-by-side.
    """
    results = {}
    for company in companies:
        docs = pipeline.retrieve(query, filters={"company_name": [company.lower()]})
        if docs:
            results[company] = docs[0].page_content[:600]
        else:
            results[company] = "No information found."

    output = []
    for company, content in results.items():
        output.append(f"### {company}\n{content}")
    return "\n\n".join(output)

# Create agent with memory
memory = InMemorySaver()
agent = create_agent(
    llm,
    tools=[get_filter_context, search_documents, compare_companies],
    checkpointer=memory,
    system_prompt=(
        "You are a senior financial analyst with access to SEC filing documents. "
        "Always call get_filter_context first to understand available data before searching. "
        "Be precise, cite specific numbers, and acknowledge uncertainty when data is missing."
    )
)

def chat(thread_id: str = "analyst-1"):
    """Interactive SEC filing analysis session."""
    config = {"configurable": {"thread_id": thread_id}}
    print("SEC Filing Analyzer — type 'quit' to exit\n")

    while True:
        query = input("You: ").strip()
        if query.lower() in ("quit", "exit"):
            break

        response = agent.invoke(
            {"messages": [HumanMessage(content=query)]},
            config=config
        )
        print(f"\nAnalyst: {response['messages'][-1].content}\n")

if __name__ == "__main__":
    chat()
```

---

## Example Queries to Demo

```
You: What was Apple's total revenue in 2024?
You: What are the main risk factors mentioned in Apple's latest 10-K?
You: Compare Apple and Microsoft's R&D spending
You: What guidance did Apple give for Q1 2025?
```

---

## Key RAGWire Concepts Covered

| Concept | Code location |
|---------|---------------|
| `get_filter_context()` | Agent-first metadata discovery |
| `retrieve(query, filters)` | Structured metadata filtering |
| Two-tool agent pattern | `get_filter_context` + `search_documents` |
| Multi-turn memory | `InMemorySaver` with `thread_id` |

---

## What to Explain in Video

1. Why agents need two tools (context + search) not one (5 min)
2. `get_filter_context()` internals — what it returns (5 min)
3. LangChain `@tool` decorator and docstring importance (5 min)
4. `create_agent` vs custom LangGraph graph (5 min)
5. `InMemorySaver` for persistent conversation threads (3 min)
6. Tool chaining — how the agent plans multi-step queries (5 min)
7. Live demo with Apple 10-K (7 min)
