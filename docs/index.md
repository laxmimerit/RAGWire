# RAGWire

**Production-grade RAG toolkit for document ingestion and retrieval with hybrid search support.**

RAGWire handles the full RAG pipeline — from loading raw documents to storing and retrieving them from a vector database — so you can focus on building your application.

---

## Features

- **Document Loading** — PDF, DOCX, XLSX, PPTX and more via MarkItDown
- **LLM Metadata Extraction** — extracts company, doc type, and fiscal period automatically
- **Smart Text Splitting** — markdown-aware and recursive chunking strategies
- **Multiple Embedding Providers** — Ollama, OpenAI, HuggingFace, Google, FastEmbed
- **Qdrant Vector Store** — dense, sparse, and hybrid search
- **Advanced Retrieval** — similarity, MMR, and hybrid search
- **SHA256 Deduplication** — at both file and chunk level — no duplicate ingestion

---

## Installation

```bash
pip install ragwire

# With Ollama support (local, no API key)
pip install "ragwire[ollama]"

# With all providers
pip install "ragwire[all]"
```

---

## Quick Start

```python
from ragwire import RAGWire

rag = RAGWire("config.yaml")

# Ingest documents
stats = rag.ingest_documents(["data/Apple_10k_2025.pdf"])
print(f"Chunks created: {stats['chunks_created']}")

# Retrieve
results = rag.retrieve("What is Apple's total revenue?", top_k=5)
for doc in results:
    print(doc.metadata.get("company_name"), doc.page_content[:200])
```

---

## Supported Providers

| Provider | Embeddings | LLM | Free |
|---|---|---|---|
| [Ollama](ollama.md) | Yes | Yes | Yes (local) |
| [OpenAI](openai.md) | Yes | Yes | No |
| [Google Gemini](gemini.md) | Yes | Yes | Free tier |
| [Groq](groq.md) | No | Yes | Free tier |
| [Anthropic](anthropic.md) | No | Yes | No |
| [HuggingFace](huggingface.md) | Yes | No | Yes (local) |
| [FastEmbed](fastembed.md) | Yes | No | Yes (local) |

---

## Links

- **GitHub**: [github.com/laxmimerit/RAGWire](https://github.com/laxmimerit/RAGWire)
- **PyPI**: [pypi.org/project/ragwire](https://pypi.org/project/ragwire)
- **YouTube**: [youtube.com/kgptalkie](https://youtube.com/kgptalkie)
- **Website**: [kgptalkie.com](https://kgptalkie.com)
