# RAGWire YouTube Tutorial Series — 15 Agentic Applications

> Build production-grade AI agents with RAGWire + LangChain, LangGraph, CrewAI, Chainlit, Streamlit, and more.

## Series Overview

| # | Title | Framework | Application |
|---|-------|-----------|-------------|
| 01 | [Financial Doc Q&A App](01_streamlit_financial_qa.md) | Streamlit | SEC filing question answering |
| 02 | [Multi-Doc Research Assistant](02_streamlit_research_assistant.md) | Streamlit | Research paper explorer with metadata filters |
| 03 | [Conversational RAG Chatbot](03_chainlit_rag_chatbot.md) | Chainlit | Streaming chat with document memory |
| 04 | [SEC Filing Analyzer Agent](04_langchain_agent_sec.md) | LangChain | Tool-calling agent over financial filings |
| 05 | [LangGraph RAG Pipeline](05_langgraph_rag_pipeline.md) | LangGraph | Stateful document ingestion + retrieval graph |
| 06 | [Self-Correcting RAG Agent](06_langgraph_self_correcting.md) | LangGraph | Agentic RAG with grading + reflection loop |
| 07 | [Multi-Agent Research Team](07_crewai_research_team.md) | CrewAI | 3-agent crew: Researcher, Analyst, Writer |
| 08 | [Investment Report Generator](08_crewai_investment_report.md) | CrewAI | Automated equity research report pipeline |
| 09 | [Autonomous Doc Intelligence Agent](09_deepagent_doc_intelligence.md) | DeepAgent | Fully autonomous document analysis loop |
| 10 | [Customer Support Knowledge Base](10_langchain_customer_support.md) | LangChain | Support bot with escalation logic |
| 11 | [Competitive Intelligence Dashboard](11_crewai_streamlit_competitive.md) | CrewAI + Streamlit | Multi-agent competitive analysis UI |
| 12 | [Interactive Workflow Builder](12_langgraph_streamlit_workflow.md) | LangGraph + Streamlit | Visual agentic pipeline with human-in-the-loop |
| 13 | [Multi-User Document Chat](13_chainlit_multiuser_chat.md) | Chainlit | Authenticated multi-user chat with per-user RAG |
| 14 | [Supervisor Multi-Agent System](14_langgraph_supervisor.md) | LangGraph | Hierarchical supervisor routing multiple specialist agents |
| 15 | [Full-Stack RAG App](15_fastapi_production_app.md) | FastAPI + Streamlit | Production deployment with API backend |

---

## Prerequisites

- Python 3.10+
- Python 3.11+
- RAGWire installed: `pip install ragwire`
- Qdrant running locally: `docker run -p 6333:6333 qdrant/qdrant`
- Ollama running locally (or OpenAI API key)
- Framework-specific installs listed in each tutorial

## Package Versions (as of March 2026)

| Package | Version |
|---------|---------|
| `langchain` | 1.2.13 |
| `langgraph` | 1.1.3 |
| `langgraph-checkpoint` | 4.0.1 |
| `langchain-ollama` | 1.0.1 |
| `crewai` | 1.11.1 |
| `chainlit` | 2.10.0 |

## Key API Notes

- Agent creation: `from langchain.agents import create_agent` (`create_react_agent` from `langgraph.prebuilt` is deprecated)
- In-memory checkpointer: `from langgraph.checkpoint.memory import InMemorySaver` (`MemorySaver` renamed)
- Human-in-the-loop: `from langgraph.types import interrupt` (unchanged)

## Common Config (`config.yaml`)

```yaml
embeddings:
  provider: "ollama"
  model: "qwen3-embedding:0.6b"
  base_url: "http://localhost:11434"

llm:
  provider: "ollama"
  model: "qwen2.5:7b"

vectorstore:
  url: "http://localhost:6333"
  collection_name: "docs"
  use_sparse: true

retriever:
  search_type: "hybrid"
  top_k: 5
  auto_filter: false
```
