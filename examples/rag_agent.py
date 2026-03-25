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
