# Build a Metadata-Aware Filtered Chatbot

Combine auto-filter with a metadata-aware system prompt so the agent knows what data is available and can filter precisely.

```python
from ragwire import RAGWire
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

rag = RAGWire("config.yaml")

# Discover what's in the collection
values = rag.get_field_values(["company_name", "doc_type", "fiscal_year"])

SYSTEM_PROMPT = f"""
You are a financial document assistant.
Always use search_documents to retrieve information before answering — never answer from general knowledge.
If no relevant documents are found, say so — do not guess or fabricate an answer.
Always cite the source document in your answer.

Available data:
- Companies  : {values['company_name']}
- Doc types  : {values['doc_type']}
- Fiscal years: {values['fiscal_year']}

The retrieval system automatically filters by company, year, and doc type
when mentioned in the query — you don't need to do anything special.
"""

@tool
def search_documents(query: str) -> str:
    """Search the financial document knowledge base."""
    results = rag.retrieve(query, top_k=5)
    if not results:
        return "No relevant documents found."
    chunks = []
    for doc in results:
        source = doc.metadata.get("file_name", "unknown")
        chunks.append(f"[{source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(chunks)

agent = create_agent(
    model=ChatOllama(model="qwen3.5:9b", base_url="http://localhost:11434"),
    tools=[search_documents],
    system_prompt=SYSTEM_PROMPT,
)

response = agent.invoke({
    "messages": [HumanMessage("What is Apple's net income for 2025?")]
})
print(response["messages"][-1].content)
```

## Why This Works

1. `get_field_values()` fetches the actual values stored in your collection — no hardcoding
2. The system prompt tells the LLM exactly what companies, years, and doc types are available
3. When the user asks "Apple's 2025 revenue", `retrieve()` auto-filters to `{company_name: "apple", fiscal_year: 2025}` — the agent doesn't need to pass filters manually

## Add Multi-Turn Memory

```python
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()

agent = create_agent(
    model=ChatOllama(model="qwen3.5:9b", base_url="http://localhost:11434"),
    tools=[search_documents],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer,
)

config = {"configurable": {"thread_id": "session-1"}}

# The agent remembers previous turns within the same thread_id
response = agent.invoke(
    {"messages": [HumanMessage("What is Apple's revenue?")]},
    config=config,
)
response = agent.invoke(
    {"messages": [HumanMessage("How does that compare to Microsoft?")]},
    config=config,
)
```

See [RAG Agent](../rag_agent.md) for the full guide including structured output and a complete working example.
