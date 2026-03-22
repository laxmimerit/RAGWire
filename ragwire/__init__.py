"""
RAGWire — Production-grade RAG toolkit for document ingestion and retrieval.

A clean, installable Python toolkit providing:
- Document loading and conversion (PDF, DOCX, XLSX, etc.)
- Text splitting and chunking
- LLM-based metadata extraction
- Multiple embedding provider support (OpenAI, HuggingFace, Ollama, Google)
- Qdrant vector store with hybrid search
- Advanced retrieval strategies (similarity, MMR, hybrid)

Example:
    >>> from ragwire import RAGPipeline
    >>>
    >>> pipeline = RAGPipeline("config.yaml")
    >>> stats = pipeline.ingest_documents(["doc.pdf"])
    >>> results = pipeline.retrieve("What is the revenue?")
    >>> for doc in results:
    ...     print(doc.page_content)

Author: KGP Talkie Private Limited
License: MIT
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("ragwire")
except PackageNotFoundError:
    __version__ = "1.0.0"

__author__ = "KGP Talkie Private Limited"

from .core.config import Config
from .core.pipeline import RAGPipeline
from .metadata.schema import DocumentMetadata
from .metadata.extractor import MetadataExtractor
from .loaders.markitdown_loader import MarkItDownLoader
from .processing.splitter import get_splitter, get_markdown_splitter, get_code_splitter
from .processing.hashing import sha256_text, sha256_file_from_path, sha256_chunk
from .embeddings.factory import get_embedding
from .vectorstores.qdrant_store import QdrantStore
from .retriever.hybrid import get_retriever, hybrid_search, mmr_search
from .utils.logging import setup_logging, get_logger

__all__ = [
    # Core
    "Config",
    "RAGPipeline",
    # Metadata
    "DocumentMetadata",
    "MetadataExtractor",
    # Loaders
    "MarkItDownLoader",
    # Processing
    "get_splitter",
    "get_markdown_splitter",
    "get_code_splitter",
    "sha256_text",
    "sha256_file_from_path",
    "sha256_chunk",
    # Embeddings
    "get_embedding",
    # Vector Stores
    "QdrantStore",
    # Retrieval
    "get_retriever",
    "hybrid_search",
    "mmr_search",
    # Utilities
    "setup_logging",
    "get_logger",
]
