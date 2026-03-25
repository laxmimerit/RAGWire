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
