# Build a Metadata-Aware Filtered Chatbot

The agent is told what companies, doc types, and fiscal years exist in the collection — then it decides which filters to pass when calling the search tool. This gives the agent full control over retrieval precision.

```python
from typing import Optional
from ragwire import RAGWire
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

rag = RAGWire("config.yaml")

# Dynamically discover all filterable fields and their values
fields = rag.get_metadata_keys()
values = rag.get_field_values(fields)
filter_context = "\n".join(f"- {field}: {vals}" for field, vals in values.items() if vals)

SYSTEM_PROMPT = f"""
You are a financial document assistant.
Always use search_documents to retrieve information before answering — never answer from general knowledge.
If no relevant documents are found, say so — do not guess or fabricate an answer.
Always cite the source document in your answer.

Available data in the knowledge base:
{filter_context}

When calling search_documents, pass the appropriate filters based on what the user is asking about.
Match filter values exactly as shown above.
Only pass filters that are clearly relevant — omit filters when the query is broad.
"""

@tool
def search_documents(query: str, filters: Optional[dict] = None) -> str:
    """
    Search the financial document knowledge base.

    Args:
        query: The search query.
        filters: Optional metadata filters (e.g. {"company_name": "apple inc.", "fiscal_year": 2025}).
    """
    results = rag.retrieve(query, top_k=5, filters=filters)
    if not results:
        return "No relevant documents found."
    chunks = []
    for doc in results:
        source = doc.metadata.get("file_name", "unknown")
        chunks.append(f"[{source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(chunks)

agent = create_agent(
    model=ChatGoogleGenerativeAI(model="gemini-2.5-flash"),
    tools=[search_documents],
    system_prompt=SYSTEM_PROMPT,
)

response = agent.invoke({
    "messages": [HumanMessage("What is Apple's net income for 2025?")]
})
print(response["messages"][-1].text)
```

## Why This Works

1. `get_field_values()` fetches the actual values stored in your collection — no hardcoding
2. The system prompt tells the agent exactly what companies, years, and doc types exist
3. The agent reads the query, decides which filters apply, and passes them explicitly to `search_documents`
4. `retrieve()` applies those filters directly — no guessing, no LLM-based auto-extraction

This is more reliable than `auto_filter` because the agent reasons about filters using the full conversation context, not just the current query string.

## Add Multi-Turn Memory

```python
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()

agent = create_agent(
    model=ChatGoogleGenerativeAI(model="gemini-2.5-flash"),
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
print(response["messages"][-1].text)
```

See [RAG Agent](../rag_agent.md) for the full guide including structured output and a complete working example.
