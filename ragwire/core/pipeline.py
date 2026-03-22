"""
Main RAG pipeline orchestrating document ingestion and retrieval.

Coordinates all components of the RAG system:
- Document loading and conversion
- Text splitting and chunking
- Metadata extraction
- Embedding generation
- Vector store operations
- Hybrid retrieval
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

# Import pipeline components
from .config import Config
from ..loaders.markitdown_loader import MarkItDownLoader
from ..processing.splitter import get_splitter, get_markdown_splitter
from ..processing.hashing import sha256_file_from_path, sha256_chunk
from ..metadata.extractor import MetadataExtractor
from ..embeddings.factory import get_embedding
from ..vectorstores.qdrant_store import QdrantStore
from ..retriever.hybrid import get_retriever, hybrid_search

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Main RAG pipeline for document ingestion and retrieval.

    Orchestrates the complete RAG workflow from document loading
    to vector store ingestion and retrieval.

    Attributes:
        config: Configuration dictionary
        loader: Document loader instance
        splitter: Text splitter instance
        embedding: Embedding model instance
        vectorstore: Qdrant vector store instance
        retriever: Retriever instance

    Example:
        >>> pipeline = RAGPipeline("config.yaml")
        >>> pipeline.ingest_documents(["doc1.pdf", "doc2.pdf"])
        >>> results = pipeline.retrieve("What is Amazon's revenue?")
    """

    def __init__(self, config_path: str):
        """
        Initialize the RAG pipeline.

        Args:
            config_path: Path to configuration YAML file

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If configuration is invalid
        """
        logger.info(f"Loading configuration from {config_path}")

        # Load configuration
        self.config_obj = Config(config_path)
        self.config = self.config_obj.config

        # Initialize components
        self._initialize_logging()
        self._initialize_loader()
        self._initialize_splitter()
        self._initialize_embeddings()
        self._initialize_llm()
        self._initialize_vectorstore()
        self._initialize_retriever()

        logger.info("RAG pipeline initialized successfully")

    def _initialize_logging(self) -> None:
        """Apply logging configuration from config file."""
        log_config = self.config.get("logging", {})
        if not log_config:
            return
        from ..utils.logging import setup_logging, setup_colored_logging
        level = log_config.get("level", "INFO")
        log_file = log_config.get("log_file")
        if log_config.get("colored", False):
            setup_colored_logging(log_level=level, log_file=log_file)
        else:
            setup_logging(
                log_level=level,
                log_file=log_file,
                console_output=log_config.get("console_output", True),
            )

    def _initialize_loader(self) -> None:
        """Initialize document loader."""
        loader_config = self.config.get("loader", {})
        self.loader = MarkItDownLoader()
        self.loader_extensions = loader_config.get(
            "extensions", [".pdf", ".docx", ".xlsx", ".pptx", ".txt", ".md"]
        )
        logger.info("Document loader initialized")

    def _initialize_splitter(self) -> None:
        """Initialize text splitter."""
        splitter_config = self.config.get("splitter", {})
        chunk_size = splitter_config.get("chunk_size", 10000)
        chunk_overlap = splitter_config.get("chunk_overlap", 2000)
        strategy = splitter_config.get("strategy", "markdown")

        if strategy == "recursive":
            self.splitter = get_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        else:
            self.splitter = get_markdown_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        logger.info(f"Text splitter initialized (strategy={strategy}, chunk_size={chunk_size})")

    def _initialize_embeddings(self) -> None:
        """Initialize embedding model."""
        embedding_config = self.config.get("embeddings", {})
        self.embedding = get_embedding(embedding_config)
        logger.info(
            f"Embedding model initialized (provider={embedding_config.get('provider')})"
        )

    def _initialize_llm(self) -> None:
        """Initialize LLM and metadata extractor."""
        llm_config = self.config.get("llm", {})
        if not llm_config:
            raise ValueError("No [llm] section found in config — required for metadata extraction")

        provider = llm_config.get("provider", "ollama")
        model = llm_config.get("model")
        if not model:
            raise ValueError("llm.model must be set in config")
        base_url = llm_config.get("base_url", "http://localhost:11434")

        temperature = llm_config.get("temperature", 0.0)

        if provider == "ollama":
            from langchain_ollama import ChatOllama
            num_ctx = llm_config.get("num_ctx", 16384)
            llm = ChatOllama(model=model, base_url=base_url, temperature=temperature, num_ctx=num_ctx)
        elif provider == "openai":
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model=model, temperature=temperature)
        elif provider == "google" or provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(model=model, temperature=temperature, google_api_key=llm_config.get("api_key"))
        elif provider == "groq":
            from langchain_groq import ChatGroq
            llm = ChatGroq(model=model, temperature=temperature, groq_api_key=llm_config.get("api_key"))
        elif provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            llm = ChatAnthropic(model=model, temperature=temperature, anthropic_api_key=llm_config.get("api_key"))
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

        self.metadata_extractor = MetadataExtractor(llm)
        logger.info(f"LLM initialized for metadata extraction (provider={provider}, model={model})")

    def _initialize_vectorstore(self) -> None:
        """Initialize vector store."""
        vectorstore_config = self.config.get("vectorstore", {})
        collection_name = vectorstore_config.get("collection_name", "rag_documents")
        use_sparse = vectorstore_config.get("use_sparse", True)
        force_recreate = vectorstore_config.get("force_recreate", False)

        self.vectorstore_wrapper = QdrantStore(
            config=vectorstore_config,
            embedding=self.embedding,
            collection_name=collection_name,
        )

        # Handle collection creation / recreation
        collection_exists = self.vectorstore_wrapper.collection_exists()

        if force_recreate and collection_exists:
            self.vectorstore_wrapper.delete_collection()
            logger.info(f"Deleted existing collection for recreation: {collection_name}")
            collection_exists = False

        if not collection_exists:
            self.vectorstore_wrapper.create_collection(use_sparse=use_sparse)
            logger.info(f"Created new collection: {collection_name}")
        else:
            logger.info(f"Using existing collection: {collection_name}")

        self.vectorstore = self.vectorstore_wrapper.get_store(use_sparse=use_sparse)
        logger.info("Vector store initialized")

    def _initialize_retriever(self) -> None:
        """Initialize retriever."""
        retriever_config = self.config.get("retriever", {})
        search_type = retriever_config.get("search_type", "hybrid")
        top_k = retriever_config.get("top_k", 5)

        self.retriever = get_retriever(
            self.vectorstore, top_k=top_k, search_type=search_type
        )
        logger.info(f"Retriever initialized (type={search_type}, top_k={top_k})")

    def ingest_documents(self, file_paths: List[str]) -> Dict[str, Any]:
        """
        Ingest documents into the vector store.

        Metadata is extracted from each document using the configured LLM.

        Args:
            file_paths: List of file paths to ingest

        Returns:
            Dictionary with ingestion statistics

        Example:
            >>> stats = pipeline.ingest_documents(["doc1.pdf", "doc2.pdf"])
            >>> print(f"Processed {stats['processed']} documents")
        """
        stats = {
            "total": len(file_paths),
            "processed": 0,
            "skipped": 0,
            "failed": 0,
            "chunks_created": 0,
            "errors": [],
        }

        logger.info(f"Starting ingestion of {len(file_paths)} documents")

        for file_path in file_paths:
            try:
                # File-level deduplication — skip if already ingested
                file_hash = sha256_file_from_path(file_path)
                if self.vectorstore_wrapper.file_hash_exists(file_hash):
                    logger.info(f"Skipping (already ingested): {file_path}")
                    stats["skipped"] += 1
                    continue

                # Load document
                result = self.loader.load(file_path)

                if not result["success"]:
                    stats["failed"] += 1
                    stats["errors"].append(
                        {"file": file_path, "error": result["error"]}
                    )
                    logger.error(f"Failed to load {file_path}: {result['error']}")
                    continue

                # Process document (pass pre-computed hash to avoid re-reading file)
                chunks = self._process_document(
                    text=result["text_content"],
                    file_path=file_path,
                    file_name=result["file_name"],
                    file_type=result["file_type"],
                    file_hash=file_hash,
                )

                # Add to vector store
                if chunks:
                    self.vectorstore.add_documents(chunks)
                    stats["chunks_created"] += len(chunks)
                    stats["processed"] += 1
                    logger.info(f"Processed {file_path}: {len(chunks)} chunks")

            except Exception as e:
                stats["failed"] += 1
                stats["errors"].append({"file": file_path, "error": str(e)})
                logger.error(f"Error processing {file_path}: {e}", exc_info=True)

        logger.info(
            f"Ingestion complete: {stats['processed']}/{stats['total']} documents"
        )
        return stats

    def _process_document(
        self,
        text: str,
        file_path: str,
        file_name: str,
        file_type: str,
        file_hash: str,
    ) -> List[Any]:
        """
        Process a single document into chunks with LLM-extracted metadata.

        Metadata is extracted once from the first chunk using the LLM,
        then attached to every chunk of the document.

        Args:
            text: Document text content
            file_path: Original file path
            file_name: Original file name
            file_type: File type
            file_hash: Pre-computed SHA256 hash of the file

        Returns:
            List of Document objects with metadata
        """
        from langchain_core.documents import Document

        # Split first so we can pass the first chunk to the LLM
        chunk_texts = self.splitter.split_text(text)

        # Extract metadata once from the first chunk using LLM
        llm_metadata = {}
        if chunk_texts:
            try:
                llm_metadata = self.metadata_extractor.extract(chunk_texts[0])
                logger.debug(f"LLM metadata for {file_name}: {llm_metadata}")
            except Exception as e:
                logger.warning(f"LLM metadata extraction failed for {file_name}: {e}")

        documents = []
        for i, chunk_text in enumerate(chunk_texts):
            chunk_id = f"{file_hash}_{i}"
            chunk_hash = sha256_chunk(chunk_id, chunk_text)

            chunk_metadata = {
                "source": file_path,
                "file_name": file_name,
                "file_type": file_type,
                "file_hash": file_hash,
                "chunk_id": chunk_id,
                "chunk_hash": chunk_hash,
                "chunk_index": i,
                "total_chunks": len(chunk_texts),
                **llm_metadata,
            }

            documents.append(Document(page_content=chunk_text, metadata=chunk_metadata))

        return documents

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        """
        Retrieve documents for a query.

        Args:
            query: Search query string
            top_k: Number of results (uses config default if not provided)
            filters: Optional metadata filters

        Returns:
            List of retrieved documents

        Example:
            >>> results = pipeline.retrieve("Amazon Q1 2024 revenue")
            >>> for doc in results:
            ...     print(doc.page_content)
        """
        if top_k is None:
            top_k = self.config.get("retriever", {}).get("top_k", 5)

        # Build search kwargs without mutating the shared retriever
        search_kwargs = {**self.retriever.search_kwargs, "k": top_k}
        if filters:
            search_kwargs["filter"] = self._build_qdrant_filter(filters)

        retriever = self.vectorstore.as_retriever(
            search_type=self.retriever.search_type,
            search_kwargs=search_kwargs,
        )
        results = retriever.invoke(query)
        logger.info(f"Retrieved {len(results)} documents for query: {query[:50]}...")

        return results

    def hybrid_search(
        self, query: str, k: int = 5, filters: Optional[Dict[str, Any]] = None
    ) -> List[Any]:
        """
        Perform hybrid search (dense + sparse).

        Args:
            query: Search query
            k: Number of results
            filters: Optional metadata filters

        Returns:
            List of retrieved documents
        """
        qdrant_filter = self._build_qdrant_filter(filters) if filters else None
        return hybrid_search(self.vectorstore, query, k=k, filters=qdrant_filter)

    @staticmethod
    def _build_qdrant_filter(filters: Dict[str, Any]) -> Any:
        """Convert a plain dict of metadata filters to a Qdrant Filter object."""
        from qdrant_client.http import models as rest

        conditions = []
        for key, value in filters.items():
            if isinstance(value, list):
                for v in value:
                    conditions.append(
                        rest.FieldCondition(
                            key=f"metadata.{key}",
                            match=rest.MatchValue(value=v),
                        )
                    )
            else:
                conditions.append(
                    rest.FieldCondition(
                        key=f"metadata.{key}",
                        match=rest.MatchValue(value=value),
                    )
                )
        return rest.Filter(must=conditions)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get pipeline statistics.

        Returns:
            Dictionary with pipeline statistics
        """
        collection_info = self.vectorstore_wrapper.get_collection_info()

        vectors = collection_info.config.params.vectors
        if hasattr(vectors, "size"):
            vector_size = vectors.size
        else:
            # Named vectors — take the first one
            vector_size = next(iter(vectors.values())).size

        return {
            "collection_name": self.vectorstore_wrapper.collection_name,
            "total_documents": collection_info.points_count or 0,
            "vector_size": vector_size,
            "indexed": getattr(collection_info, "indexed_vectors_count", None) or 0,
        }
