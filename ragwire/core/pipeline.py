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

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any, TypedDict, Iterable, Iterator


class IngestError(TypedDict):
    file: str
    error: str


class IngestStats(TypedDict):
    total: int
    processed: int
    skipped: int
    failed: int
    chunks_created: int
    #: Documents that were ingested but whose LLM metadata extraction failed.
    #: These are counted in `processed` — the text is searchable, but they will
    #: not match any metadata filter until re-ingested.
    metadata_failed: int
    #: Documents whose content changed since a previous ingest, where the older
    #: version's chunks were removed before writing the new one.
    replaced: int
    errors: List[IngestError]

from langchain_core.prompts import ChatPromptTemplate

# Import pipeline components
from .config import Config
from ..loaders.markitdown_loader import MarkItDownLoader
from ..processing.splitter import get_splitter, get_markdown_splitter
from ..processing.hashing import sha256_file_from_path, sha256_chunk, sha256_text
from ..utils.retry import retry_call
from ..metadata.extractor import MetadataExtractor
from ..metadata.schema import DocumentMetadata
from ..embeddings.factory import get_embedding
from ..vectorstores.qdrant_store import QdrantStore
from ..retriever.hybrid import get_retriever, hybrid_search

logger = logging.getLogger(__name__)


class RAGWire:
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
        >>> rag = RAGWire("config.yaml")
        >>> rag.ingest_documents(["doc1.pdf", "doc2.pdf"])
        >>> results = rag.retrieve("What is Amazon's revenue?")
    """

    # Ingestion defaults. Set from config in _initialize_ingestion(); declared
    # here so the values are always present and conservative.
    _workers = 1
    _batch_size = 64
    _write_retries = 2
    _replace_changed = True
    _dedup_chunks = False
    _metadata_retries = 2

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
        self.config = Config(config_path).config

        # Cache for stored filter values — populated on first query, invalidated after ingestion
        self._stored_values_cache: Optional[Dict[str, Any]] = None

        # Initialize components
        self._initialize_logging()
        self._initialize_loader()
        self._initialize_splitter()
        self._initialize_ingestion()
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

    def _initialize_ingestion(self) -> None:
        """Read the optional [ingestion] config block."""
        cfg = self.config.get("ingestion", {}) or {}

        self._workers = max(1, int(cfg.get("workers", 1)))
        self._batch_size = max(1, int(cfg.get("batch_size", 64)))
        self._write_retries = max(0, int(cfg.get("retries", 2)))
        self._replace_changed = bool(cfg.get("replace_changed", True))
        self._dedup_chunks = bool(cfg.get("dedup_chunks", False))

        if self._workers > 1:
            logger.info(
                f"Ingestion will prepare up to {self._workers} documents "
                "concurrently (loading, splitting and metadata extraction). "
                "Vector store writes stay sequential."
            )
        logger.info(
            f"Ingestion configured (workers={self._workers}, "
            f"batch_size={self._batch_size}, retries={self._write_retries}, "
            f"replace_changed={self._replace_changed})"
        )

    def _initialize_embeddings(self) -> None:
        """Initialize embedding model."""
        embedding_config = self.config.get("embeddings", {})
        if not embedding_config or not embedding_config.get("provider"):
            raise ValueError(
                "Missing [embeddings] section or embeddings.provider in config.yaml.\n"
                "Example:\n"
                "  embeddings:\n"
                "    provider: ollama\n"
                "    model: nomic-embed-text\n"
                "Valid providers: ollama, openai, openrouter, huggingface, google, fastembed"
            )
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

        _llm_install = {
            "ollama": "pip install langchain-ollama",
            "openai": "pip install \"ragwire[openai]\"",
            "openrouter": "pip install \"ragwire[openrouter]\"",
            "google": "pip install \"ragwire[google]\"",
            "gemini": "pip install \"ragwire[google]\"",
            "groq": "pip install \"ragwire[groq]\"",
            "anthropic": "pip install \"ragwire[anthropic]\"",
        }
        try:
            if provider == "ollama":
                from langchain_ollama import ChatOllama
                extra = {}
                if "num_ctx" in llm_config:
                    extra["num_ctx"] = llm_config["num_ctx"]
                llm = ChatOllama(model=model, base_url=base_url, **extra)
            elif provider == "openai":
                from langchain_openai import ChatOpenAI
                llm = ChatOpenAI(model=model)
            elif provider == "openrouter":
                # Use the dedicated ChatOpenRouter integration, NOT ChatOpenAI with a
                # base_url override. The override approach breaks with_structured_output()
                # — which the metadata extractor relies on. ChatOpenRouter supports
                # native structured output and tool calling.
                # Reads the OPENROUTER_API_KEY env var when api_key is not set in config.
                from langchain_openrouter import ChatOpenRouter
                extra = {}
                if llm_config.get("api_key"):
                    extra["api_key"] = llm_config["api_key"]
                llm = ChatOpenRouter(model=model, **extra)
            elif provider == "google" or provider == "gemini":
                from langchain_google_genai import ChatGoogleGenerativeAI
                llm = ChatGoogleGenerativeAI(model=model, google_api_key=llm_config.get("api_key"))
            elif provider == "groq":
                from langchain_groq import ChatGroq
                llm = ChatGroq(model=model, groq_api_key=llm_config.get("api_key"))
            elif provider == "anthropic":
                from langchain_anthropic import ChatAnthropic
                llm = ChatAnthropic(model=model, anthropic_api_key=llm_config.get("api_key"))
            else:
                valid = "ollama, openai, openrouter, google, groq, anthropic"
                raise ValueError(
                    f"Unsupported LLM provider: '{provider}'. Valid options: {valid}"
                )
        except ImportError:
            install_cmd = _llm_install.get(provider, f"pip install \"ragwire[{provider}]\"")
            raise ImportError(
                f"Required package for LLM provider '{provider}' is not installed.\n"
                f"Run: {install_cmd}"
            )

        metadata_config = self.config.get("metadata", {})
        metadata_yaml = metadata_config.get("config_file") if metadata_config else None

        if metadata_yaml:
            self.metadata_extractor = MetadataExtractor.from_yaml(llm, metadata_yaml)
            logger.info(f"Metadata extractor loaded from: {metadata_yaml}")
            self._filter_fields = self.metadata_extractor.fields or ["company_name", "doc_type", "fiscal_quarter", "fiscal_year"]
        else:
            self.metadata_extractor = MetadataExtractor(llm)
            self._filter_fields = ["company_name", "doc_type", "fiscal_quarter", "fiscal_year"]

        # Payload index types come from the schema, so custom integer fields are
        # indexed as integers rather than defaulting to keyword.
        self._field_types = self.metadata_extractor.field_types

        # How many times to retry metadata extraction before giving up on a file
        self._metadata_retries = metadata_config.get("retries", 2) if metadata_config else 2
        logger.info(f"LLM initialized for metadata extraction (provider={provider}, model={model})")

    def _initialize_vectorstore(self) -> None:
        """Initialize vector store."""
        vectorstore_config = self.config.get("vectorstore", {})
        if not vectorstore_config or not vectorstore_config.get("url"):
            raise ValueError(
                "Missing [vectorstore] section or vectorstore.url in config.yaml.\n"
                "Example:\n"
                "  vectorstore:\n"
                "    url: http://localhost:6333\n"
                "    collection_name: my_docs\n"
                "Start Qdrant locally with: docker run -p 6333:6333 qdrant/qdrant"
            )
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
            self._check_embedding_dimension(collection_name)
            logger.info(f"Using existing collection: {collection_name}")

        self.vectorstore = self.vectorstore_wrapper.get_store(use_sparse=use_sparse)
        existing_fields = self.vectorstore_wrapper.get_metadata_keys()
        self.vectorstore_wrapper.create_payload_indexes(
            ["file_hash"] + existing_fields, field_types=self._field_types
        )
        logger.info("Vector store initialized")

    def _check_embedding_dimension(self, collection_name: str) -> None:
        """
        Fail fast when the configured embedding model does not match the collection.

        Attaching a 1024-dim model to a collection built with a 768-dim one
        otherwise succeeds here and blows up much later inside add_documents with
        a raw Qdrant error that names neither the model nor the fix.
        """
        stored_size = self.vectorstore_wrapper.get_vector_size()
        if stored_size is None:
            return

        current_size = len(self.embedding.embed_query("test"))
        if current_size == stored_size:
            return

        raise ValueError(
            f"Embedding dimension mismatch for collection '{collection_name}'.\n"
            f"  Collection was created with : {stored_size} dimensions\n"
            f"  Configured model produces   : {current_size} dimensions "
            f"(provider={self.config.get('embeddings', {}).get('provider')}, "
            f"model={self.config.get('embeddings', {}).get('model')})\n"
            "An embedding model can only query a collection built with the same model.\n"
            "Either restore the previous embeddings.model, or set "
            "vectorstore.force_recreate: true once to rebuild the collection "
            "(this deletes all ingested data) and re-ingest your documents."
        )

    def _initialize_retriever(self) -> None:
        """Initialize retriever."""
        retriever_config = self.config.get("retriever", {})
        search_type = retriever_config.get("search_type", "hybrid")
        top_k = retriever_config.get("top_k", 5)
        self._auto_filter = retriever_config.get("auto_filter", False)
        self.retriever = get_retriever(
            self.vectorstore, top_k=top_k, search_type=search_type
        )
        logger.info(f"Retriever initialized (type={search_type}, top_k={top_k}, auto_filter={self._auto_filter})")

    def ingest_documents(self, file_paths: List[str]) -> IngestStats:
        """
        Ingest documents into the vector store.

        Metadata is extracted from each document using the configured LLM.

        Args:
            file_paths: List of file paths to ingest

        Returns:
            Dictionary with ingestion statistics

        Example:
            >>> stats = rag.ingest_documents(["doc1.pdf", "doc2.pdf"])
            >>> print(f"Processed {stats['processed']} documents")
        """
        stats: IngestStats = {
            "total": len(file_paths),
            "processed": 0,
            "skipped": 0,
            "failed": 0,
            "chunks_created": 0,
            "metadata_failed": 0,
            "replaced": 0,
            "errors": [],
        }

        logger.info(f"Starting ingestion of {len(file_paths)} documents")

        try:
            from tqdm import tqdm
            file_iter = tqdm(file_paths, desc="Ingesting", unit="file")
        except ImportError:
            file_iter = file_paths

        # Documents are prepared concurrently (loading, splitting and the LLM
        # metadata call dominate wall-clock) but written sequentially, so stats,
        # ordering and rollback stay simple to reason about.
        for prepared in self._prepare_documents(file_iter):
            self._write_prepared(prepared, stats)

        # Create payload indexes for all metadata fields so facet API works
        all_fields = self.vectorstore_wrapper.get_metadata_keys()
        self.vectorstore_wrapper.create_payload_indexes(
            all_fields, field_types=self._field_types
        )
        self._stored_values_cache = None  # invalidate after ingestion

        logger.info(
            f"Ingestion complete: {stats['processed']}/{stats['total']} documents "
            f"({stats['skipped']} skipped, {stats['failed']} failed, "
            f"{stats['replaced']} replaced, {stats['chunks_created']} chunks)"
        )
        if stats["metadata_failed"]:
            logger.warning(
                f"{stats['metadata_failed']} document(s) were ingested without "
                "metadata — they will not match metadata filters. Re-ingest them "
                "with rag.reingest_documents([...]) once the LLM is reachable."
            )
        return stats

    def _prepare_documents(self, file_paths: Iterable[str]) -> Iterator[dict]:
        """
        Yield a prepared record per file: hash, chunks and what to do with them.

        Everything here is read-only with respect to the vector store, which is
        what makes it safe to run concurrently. With workers=1 the work happens
        inline and lazily, so a long ingest still streams progress.

        Args:
            file_paths: Paths to prepare (may be a tqdm-wrapped iterable)

        Yields:
            Dicts as returned by _prepare_document()
        """
        if self._workers == 1:
            for file_path in file_paths:
                yield self._prepare_document(file_path)
            return

        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=self._workers) as pool:
            # map keeps input order, so results are written deterministically
            yield from pool.map(self._prepare_document, file_paths)

    def _prepare_document(self, file_path: str) -> dict:
        """
        Hash, dedup-check, load, split and extract metadata for one file.

        Runs in a worker thread when workers > 1, so it must not write to the
        vector store. Failures are captured in the returned record rather than
        raised, so one bad file cannot abort the pool.

        Args:
            file_path: Path to the document

        Returns:
            Dict with keys: file, hash, chunks, metadata_ok, error, action,
            stored, expected. ``action`` is one of "skip", "write" or "error".
        """
        record = {
            "file": file_path, "hash": None, "chunks": None,
            "metadata_ok": True, "error": None, "action": "error",
            "stored": 0, "expected": None, "was_partial": False,
        }

        try:
            file_hash = sha256_file_from_path(file_path)
            record["hash"] = file_hash

            # A previous run that died partway leaves chunks behind — those must
            # be cleared and re-ingested, not mistaken for a finished document.
            status, stored, expected = self.vectorstore_wrapper.get_ingest_status(
                file_hash
            )
            record["stored"], record["expected"] = stored, expected

            if status == "complete":
                record["action"] = "skip"
                return record
            record["was_partial"] = status == "partial"

            result = self.loader.load(file_path)
            if not result["success"]:
                record["error"] = result["error"]
                return record

            chunks, metadata_ok = self._process_document(
                text=result["text_content"],
                file_path=file_path,
                file_name=result["file_name"],
                file_type=result["file_type"],
                file_hash=file_hash,
            )
            record["metadata_ok"] = metadata_ok

            if not chunks:
                # Scanned/image-only PDFs and empty files land here. Counting
                # them in neither processed nor failed makes them invisible.
                record["error"] = (
                    "no extractable text (possibly a scanned or image-only document)"
                )
                return record

            record["chunks"] = chunks
            record["action"] = "write"
            return record

        except Exception as e:
            logger.debug(f"Preparation failed for {file_path}: {e}", exc_info=True)
            record["error"] = str(e)
            return record

    def _write_prepared(self, record: dict, stats: IngestStats) -> None:
        """
        Write one prepared document to the vector store and update stats.

        Sequential by design: it owns every mutation of both the collection and
        the stats dict, so there is exactly one place where partial state can be
        created and rolled back.

        Args:
            record: A record from _prepare_document()
            stats: Stats dict mutated in place
        """
        file_path = record["file"]
        file_hash = record["hash"]

        if record["action"] == "skip":
            logger.info(f"Skipping (already ingested): {file_path}")
            stats["skipped"] += 1
            return

        if record["action"] == "error":
            stats["failed"] += 1
            stats["errors"].append({"file": file_path, "error": record["error"]})
            logger.error(f"Failed to process {file_path}: {record['error']}")
            return

        chunks = record["chunks"]
        try:
            if record["was_partial"]:
                logger.warning(
                    f"Found incomplete ingest for {file_path} "
                    f"({record['stored']}/{record['expected']} chunks) "
                    "— clearing and re-ingesting"
                )
                self.vectorstore_wrapper.delete_by_file_hash(file_hash)

            if self._replace_changed:
                # The file's content changed, so its hash changed too. Without
                # this the previous version stays in the collection alongside
                # the new one and keeps surfacing in results.
                stale = self.vectorstore_wrapper.delete_by_source(
                    file_path, except_file_hash=file_hash
                )
                if stale:
                    stats["replaced"] += 1
                    logger.info(
                        f"Replaced {stale} chunk(s) from a previous version of "
                        f"{file_path}"
                    )

            self._write_chunks(chunks, file_path)

            stats["chunks_created"] += len(chunks)
            stats["processed"] += 1
            if not record["metadata_ok"]:
                stats["metadata_failed"] += 1
            logger.info(f"Processed {file_path}: {len(chunks)} chunks")

        except Exception as e:
            stats["failed"] += 1
            stats["errors"].append({"file": file_path, "error": str(e)})
            logger.error(f"Error writing {file_path}: {e}", exc_info=True)

            # Roll back anything that landed before the failure, so the file is
            # retried cleanly next run instead of being skipped as done.
            if file_hash:
                try:
                    self.vectorstore_wrapper.delete_by_file_hash(file_hash)
                except Exception as cleanup_error:
                    logger.warning(
                        f"Could not roll back partial ingest for {file_path}: "
                        f"{cleanup_error}"
                    )

    def _write_chunks(self, chunks: List[Any], file_path: str) -> None:
        """
        Add chunks to the vector store in retried batches.

        A single add_documents call for a large document is one oversized
        request that can exceed body limits or time out; batching also means a
        transient failure retries a batch rather than the whole document.

        Args:
            chunks: Documents to write
            file_path: Used only in log messages
        """
        total_batches = (len(chunks) + self._batch_size - 1) // self._batch_size

        for index in range(total_batches):
            start = index * self._batch_size
            batch = chunks[start : start + self._batch_size]
            retry_call(
                lambda b=batch: self.vectorstore.add_documents(b),
                attempts=self._write_retries + 1,
                description=(
                    f"write batch {index + 1}/{total_batches} of {file_path}"
                ),
            )

    def ingest_directory(
        self,
        directory: str,
        recursive: bool = False,
        extensions: Optional[List[str]] = None,
    ) -> IngestStats:
        """
        Ingest all supported documents from a directory.

        Args:
            directory: Path to the directory
            recursive: Whether to search subdirectories (default: False)
            extensions: File extensions to include (defaults to loader config)

        Returns:
            Dictionary with ingestion statistics

        Example:
            >>> stats = rag.ingest_directory("data/", recursive=True)
        """
        dir_path = Path(directory)
        if not dir_path.is_dir():
            raise ValueError(f"Not a directory: {directory}")

        exts = extensions or self.loader_extensions
        pattern = "**/*" if recursive else "*"

        file_paths = [
            str(p) for p in dir_path.glob(pattern)
            if p.is_file() and p.suffix.lower() in exts
        ]

        if not file_paths:
            logger.warning(f"No supported files found in {directory} (extensions: {exts})")
            return {
                "total": 0, "processed": 0, "skipped": 0,
                "failed": 0, "chunks_created": 0, "metadata_failed": 0,
                "replaced": 0, "errors": [],
            }

        logger.info(f"Found {len(file_paths)} file(s) in {directory}")
        return self.ingest_documents(file_paths)

    def reingest_documents(self, file_paths: List[str]) -> IngestStats:
        """
        Force re-ingestion of documents, replacing anything already stored.

        Deletes every existing chunk for each file before ingesting, so this
        recovers documents that were stored without metadata (LLM was down) or
        whose content has changed on disk since the last run.

        Args:
            file_paths: List of file paths to re-ingest

        Returns:
            Dictionary with ingestion statistics

        Example:
            >>> stats = rag.reingest_documents(["report.pdf"])
        """
        for file_path in file_paths:
            try:
                file_hash = sha256_file_from_path(file_path)
                removed = self.vectorstore_wrapper.delete_by_file_hash(file_hash)
                if removed:
                    logger.info(f"Cleared {removed} existing chunk(s) for {file_path}")
            except FileNotFoundError:
                # ingest_documents reports this per-file in its error list
                continue

        return self.ingest_documents(file_paths)

    def delete_document(self, file_path: str) -> int:
        """
        Remove every chunk of a document from the collection.

        Args:
            file_path: Path to the source file (must still exist — its hash is
                       what identifies the stored chunks)

        Returns:
            Number of chunks deleted

        Example:
            >>> rag.delete_document("data/old_report.pdf")
            42
        """
        file_hash = sha256_file_from_path(file_path)
        removed = self.vectorstore_wrapper.delete_by_file_hash(file_hash)
        self._stored_values_cache = None  # stored values may have changed
        return removed

    def _process_document(
        self,
        text: str,
        file_path: str,
        file_name: str,
        file_type: str,
        file_hash: str,
    ) -> tuple:
        """
        Process a single document into chunks with LLM-extracted metadata.

        Metadata is extracted once from the document text, then attached to
        every chunk of that document.

        Args:
            text: Document text content
            file_path: Original file path
            file_name: Original file name
            file_type: File type
            file_hash: Pre-computed SHA256 hash of the file

        Returns:
            Tuple of (list of Document objects, metadata_ok) where metadata_ok is
            False if LLM extraction failed and the chunks carry no semantic metadata.
        """
        from langchain_core.documents import Document

        # Split first so an empty document short-circuits before the LLM call
        chunk_texts = self.splitter.split_text(text)
        if not chunk_texts:
            return ([], True)

        # Extract from the full document text, capped inside extract() — the
        # first chunk alone was too little context to reliably find all fields.
        llm_metadata, metadata_ok = self._extract_metadata_with_retry(text, file_name)

        if self._dedup_chunks:
            chunk_texts = self._drop_duplicate_chunks(chunk_texts, file_name)

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
                # Hash of the text alone. chunk_hash mixes in chunk_id, so it
                # differs for identical text; this one is comparable across
                # chunks and documents.
                "content_hash": sha256_text(chunk_text),
                "chunk_index": i,
                "total_chunks": len(chunk_texts),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "metadata_status": "ok" if metadata_ok else "failed",
                **llm_metadata,
            }

            # Validate the system fields before they reach the store. Custom
            # schema fields pass through untouched (extra="allow"); what this
            # catches is a custom field colliding with a system field name,
            # which would silently corrupt provenance. The payload itself is
            # stored unchanged so no nulls are introduced for absent fields.
            DocumentMetadata(**chunk_metadata)

            documents.append(Document(page_content=chunk_text, metadata=chunk_metadata))

        return (documents, metadata_ok)

    @staticmethod
    def _drop_duplicate_chunks(chunk_texts: List[str], file_name: str) -> List[str]:
        """
        Remove repeated chunks within a single document, keeping the first.

        Long filings repeat boilerplate (cover blocks, legends, disclaimers)
        across sections, and those near-identical chunks crowd out real content
        in top-k results. Comparison is exact on the stripped text, so this only
        removes genuine duplicates.

        Scope is deliberately one document: cross-document deduplication would
        need a lookup per chunk against the whole collection, which costs far
        more than it saves at ingest time.

        Args:
            chunk_texts: Chunk texts in document order
            file_name: Used only in the log message

        Returns:
            Chunk texts with later exact duplicates removed
        """
        seen = set()
        unique = []
        for text in chunk_texts:
            key = sha256_text(text.strip())
            if key in seen:
                continue
            seen.add(key)
            unique.append(text)

        dropped = len(chunk_texts) - len(unique)
        if dropped:
            logger.info(f"Dropped {dropped} duplicate chunk(s) from {file_name}")
        return unique

    def _extract_metadata_with_retry(self, text: str, file_name: str) -> tuple:
        """
        Extract metadata, retrying transient LLM failures with backoff.

        A single timeout used to be swallowed with a warning, ingesting the
        document with no semantic metadata — and because file-hash dedup then
        skipped it on every later run, that state was permanent. Retrying first,
        and tagging what still fails, makes it both rarer and recoverable.

        Args:
            text: Document text to extract from
            file_name: Name used in log messages

        Returns:
            Tuple of (metadata dict, ok) — ok is False when every attempt failed.
        """
        try:
            metadata = retry_call(
                lambda: self.extract_metadata(text),
                attempts=max(1, self._metadata_retries + 1),
                description=f"metadata extraction for {file_name}",
            )
            logger.debug(f"LLM metadata for {file_name}: {metadata}")
            return (metadata, True)
        except Exception as e:
            logger.error(
                f"Metadata extraction failed for {file_name}: {e}. "
                "Ingesting without metadata, so this document will not match "
                "metadata filters until it is re-ingested."
            )
            return ({}, False)

    @property
    def filter_fields(self) -> List[str]:
        """Return the metadata fields used for filtering and auto-filter extraction.

        These are the semantic/LLM-extracted fields only (e.g. company_name, doc_type,
        fiscal_year). System fields like file_hash, chunk_id, source are excluded.
        Use this instead of discover_metadata_fields() when building filter prompts.
        """
        return self._filter_fields

    @property
    def _stored_values(self) -> Dict[str, Any]:
        """Return cached stored filter values, fetching from Qdrant if needed."""
        if self._stored_values_cache is None:
            self._stored_values_cache = self.vectorstore_wrapper.get_field_values(
                self._filter_fields, limit=50, field_types=self._field_types
            )
        return self._stored_values_cache

    def extract_filters(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Extract metadata filters from a natural language query.

        Returns the raw extracted filters so the caller (e.g. an agent) can
        inspect, adjust, or discard them before passing to retrieve().

        Args:
            query: Natural language query string

        Returns:
            Dict of extracted filters, or None if nothing was extracted.

        Example:
            >>> filters = rag.extract_filters("muscle building studies from 2023")
            >>> # {"research_focus": "muscle building", "publication_year": 2023}
            >>> # Agent inspects and adjusts if needed
            >>> results = rag.retrieve(query, filters=filters)
        """
        return self._extract_filters_from_query(query)

    def get_filter_context(self, query: str, limit: int = 50) -> str:
        """
        Build a ready-made prompt block for an agent describing available metadata
        filters, their stored values, and the filters extracted from the current query.

        Append or prepend this to your agent's task prompt so the agent can decide
        whether to apply, adjust, or discard the extracted filters before calling retrieve().

        Args:
            query: Natural language query string
            limit: Max stored values to show per field (default: 50)

        Returns:
            Formatted markdown string ready to inject into an agent prompt.

        Example:
            >>> context = rag.get_filter_context("muscle building studies from 2023")
            >>> agent_prompt = context + "\\n\\n" + your_task_prompt
        """
        stored_values = self.get_field_values(self._filter_fields, limit=limit)
        extracted = self.extract_filters(query) or {}

        lines = ["## RAGWire Filter Context", ""]
        lines.append("### Available Metadata Fields and Stored Values")
        for field in self._filter_fields:
            values = stored_values.get(field, [])
            lines.append(f"- **{field}**: {values}")

        lines.append("")
        lines.append("### Extracted Filters from Query")
        if extracted:
            for k, v in extracted.items():
                lines.append(f"- **{k}**: `{v}`")
        else:
            lines.append("- *(no filters extracted)*")

        lines += [
            "",
            "### Instructions",
            "1. Review the extracted filters above.",
            "2. If an extracted value does not match or closely relate to any stored value, adjust or drop that filter.",
            "3. If the query has no clear metadata intent, pass an empty dict `{}` as filters.",
            "4. Pass the final filters dict to the retrieval tool as `filters=`.",
        ]

        return "\n".join(lines)

    def _extract_filters_from_query(self, query: str) -> Optional[Dict[str, Any]]:
        """Use the configured LLM to extract metadata filters from a natural language query.

        Passes actual stored values to the LLM so it can match exactly what's in
        the collection — avoids mismatches like 'apple' vs 'apple inc.'.
        """
        stored_values = self._stored_values
        fields_desc = "\n".join(
            f"  {field}: {stored_values.get(field, [])}"
            for field in self._filter_fields
        )

        prompt_template = (
            "You are a metadata filter extractor for a document retrieval system.\n\n"
            "## Task\n"
            "Extract metadata filters as a JSON object from the user query.\n"
            "The filters will be used to narrow down document search results.\n\n"
            "## Rules\n"
            "1. Extract a field only when the query clearly and explicitly refers to it.\n"
            "2. Always extract the value the user asked for — but first check if it is an alias, brand name, or subsidiary of a stored value.\n"
            "   If the extracted value refers to the same real-world entity as a stored value (e.g. 'google' → 'alphabet inc.', 'instagram' → 'meta'), use the stored value instead.\n"
            "   If no stored value matches, extract exactly what the user said.\n"
            "3. Learn the format and structure from stored values, then apply that same format to what the user asked for:\n"
            "   - Casing: if stored values are lowercase, output lowercase.\n"
            "   - Prefixes/suffixes: if stored values use a prefix (e.g. 'q1', 'v2', 'dept-hr'), apply it.\n"
            "   - Data type: if stored values are integers, output integers; if strings, output strings.\n"
            "   - Lists: if stored values are lists (e.g. [2024, 2025]), output a list.\n"
            "4. When a query asks for multiple values of the same field (e.g. '2023 and 2024'), output them as a list.\n"
            "5. Do not infer or guess filters that are not clearly mentioned in the query.\n"
            "6. Return {{}} if the query contains no metadata references at all.\n\n"
            "## Format Examples from Stored Values (not a whitelist)\n"
            f"{fields_desc}\n\n"
            "## Examples\n"
            "- Stored: fiscal_quarter: ['q1','q2','q3'] | Query: 'show me Q4 reports' → {{\"fiscal_quarter\": \"q4\"}}\n"
            "- Stored: fiscal_year: [2024, 2025]       | Query: 'documents from 2022'  → {{\"fiscal_year\": 2022}}\n"
            "- Stored: department: ['engineering']     | Query: 'HR policies'          → {{\"department\": \"hr\"}}\n"
            "- Stored: language: ['en']                | Query: 'French documents'     → {{\"language\": \"fr\"}}\n"
            "- Stored: status: ['active']              | Query: 'all documents'        → {{}}\n"
            "- Stored: company_name: ['alphabet inc.'] | Query: 'google earnings'       → {{\"company_name\": \"alphabet inc.\"}}\n\n"
            "## User Query\n"
            "{query}\n\n"
            "## Output (JSON only, no explanation)\n"
        )

        try:
            chain = ChatPromptTemplate.from_template(prompt_template) | self.metadata_extractor.llm
            response = chain.invoke({"query": query})
            text = response.text.strip()
            start = text.find("{")
            if start != -1:
                filters, _ = json.JSONDecoder().raw_decode(text, start)
                if filters:
                    filters = self._normalize_filters(filters)
                    logger.info(f"Auto-extracted filters from query: {filters}")
                    return filters
        except Exception as e:
            logger.warning(f"Auto filter extraction failed: {e}")
        return None

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
            >>> results = rag.retrieve("Amazon Q1 2024 revenue")
            >>> for doc in results:
            ...     print(doc.page_content)
        """
        if top_k is None:
            top_k = self.config.get("retriever", {}).get("top_k", 5)

        if filters is None and self._auto_filter:
            filters = self._extract_filters_from_query(query)

        # Build search kwargs without mutating the shared retriever
        search_kwargs = {**self.retriever.search_kwargs, "k": top_k}
        if filters:
            search_kwargs["filter"] = self._build_qdrant_filter(
                self._normalize_filters(filters)
            )

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
        if filters is None and self._auto_filter:
            filters = self._extract_filters_from_query(query)
        qdrant_filter = (
            self._build_qdrant_filter(self._normalize_filters(filters))
            if filters
            else None
        )
        return hybrid_search(self.vectorstore, query, k=k, filters=qdrant_filter)

    @staticmethod
    def _normalize_filters(filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Lowercase and trim string filter values to match how they are stored.

        MetadataExtractor lowercases every string on write, and Qdrant's
        MatchValue is exact and case-sensitive. Without this, a caller passing
        ``{"company_name": "Apple Inc."}`` — exactly what an LLM agent produces
        from a user's question — matches zero points against the stored
        ``"apple inc."``.

        Args:
            filters: Raw filter dict from a caller or from the LLM

        Returns:
            Filter dict with string values normalized
        """
        def _norm(value: Any) -> Any:
            if isinstance(value, str):
                return value.lower().strip()
            if isinstance(value, list):
                return [_norm(item) for item in value]
            return value

        return {k: _norm(v) for k, v in filters.items()}

    @staticmethod
    def _build_qdrant_filter(filters: Dict[str, Any]) -> Any:
        """Convert a plain dict of metadata filters to a Qdrant Filter object."""
        from qdrant_client.http import models as rest

        conditions = []
        for key, value in filters.items():
            if isinstance(value, list):
                # OR logic within a field: doc must match any one of the values
                # (e.g. fiscal_year [2023, 2024] → year is 2023 OR 2024)
                conditions.append(
                    rest.Filter(
                        should=[
                            rest.FieldCondition(
                                key=f"metadata.{key}",
                                match=rest.MatchValue(value=v),
                            )
                            for v in value
                        ]
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

    def discover_metadata_fields(self) -> List[str]:
        """
        Return all metadata field names present in the collection.

        Scrolls a single point from Qdrant to inspect its payload keys.
        Fast — one network call regardless of collection size.

        Returns:
            List of metadata field names, or empty list if collection is empty

        Example:
            >>> fields = rag.discover_metadata_fields()
            >>> print(fields)
            ['company_name', 'doc_type', 'fiscal_year', 'file_name', ...]
        """
        return self.vectorstore_wrapper.get_metadata_keys()

    def get_field_values(
        self,
        fields: Any,
        limit: int = 50,
    ) -> Any:
        """
        Return unique values for one or more metadata fields.

        Uses Qdrant's facet API — fast and exact regardless of collection size.
        Creates a payload index on each field automatically if one doesn't exist.

        Args:
            fields: A field name (str) or list of field names
            limit: Max unique values to return per field (default: 50)

        Returns:
            - If fields is a str: list of unique values for that field
            - If fields is a list: dict mapping field name → list of unique values

        Example:
            >>> rag.get_field_values("company_name")
            ['apple', 'microsoft', 'google']

            >>> rag.get_field_values(["company_name", "doc_type"])
            {'company_name': ['apple', 'microsoft'], 'doc_type': ['10-k', '10-q']}
        """
        single = isinstance(fields, str)
        field_list = [fields] if single else fields
        result = self.vectorstore_wrapper.get_field_values(
            field_list, limit=limit, field_types=self._field_types
        )
        return result[fields] if single else result

    def extract_metadata(self, text: str) -> Dict[str, Any]:
        """
        Extract metadata from text using the configured LLM.

        Automatically passes stored collection values so the LLM reuses
        existing entity names (e.g. 'apple inc.') instead of extracting
        inconsistent variants ('apple', 'Apple Inc.').

        Args:
            text: Document text to extract metadata from

        Returns:
            Dictionary of extracted metadata fields

        Example:
            >>> metadata = rag.extract_metadata(open("report.pdf.txt").read())
            >>> print(metadata)
            {'company_name': 'apple inc.', 'doc_type': '10-k', 'fiscal_year': [2025]}
        """
        return self.metadata_extractor.extract(text, stored_values=self._stored_values)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get pipeline statistics.

        Returns:
            Dictionary with pipeline statistics
        """
        collection_info = self.vectorstore_wrapper.get_collection_info()

        return {
            "collection_name": self.vectorstore_wrapper.collection_name,
            "total_documents": collection_info.points_count or 0,
            "vector_size": self.vectorstore_wrapper.get_vector_size(),
            "indexed": getattr(collection_info, "indexed_vectors_count", None) or 0,
        }
