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
from typing import Optional, List, Dict, Any

from langchain_core.prompts import ChatPromptTemplate

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

        if provider == "ollama":
            from langchain_ollama import ChatOllama
            extra = {}
            if "num_ctx" in llm_config:
                extra["num_ctx"] = llm_config["num_ctx"]
            llm = ChatOllama(model=model, base_url=base_url, **extra)
        elif provider == "openai":
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model=model)
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
            raise ValueError(f"Unsupported LLM provider: {provider}")

        metadata_config = self.config.get("metadata", {})
        metadata_yaml = metadata_config.get("config_file") if metadata_config else None

        if metadata_yaml:
            self.metadata_extractor = MetadataExtractor.from_yaml(llm, metadata_yaml)
            logger.info(f"Metadata extractor loaded from: {metadata_yaml}")
            self._filter_fields = self.metadata_extractor.fields or ["company_name", "doc_type", "fiscal_quarter", "fiscal_year"]
        else:
            self.metadata_extractor = MetadataExtractor(llm)
            self._filter_fields = ["company_name", "doc_type", "fiscal_quarter", "fiscal_year"]
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
        self._auto_filter = retriever_config.get("auto_filter", False)
        self.retriever = get_retriever(
            self.vectorstore, top_k=top_k, search_type=search_type
        )
        logger.info(f"Retriever initialized (type={search_type}, top_k={top_k}, auto_filter={self._auto_filter})")

    def ingest_documents(self, file_paths: List[str]) -> Dict[str, Any]:
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
        stats = {
            "total": len(file_paths),
            "processed": 0,
            "skipped": 0,
            "failed": 0,
            "chunks_created": 0,
            "errors": [],
        }

        logger.info(f"Starting ingestion of {len(file_paths)} documents")

        try:
            from tqdm import tqdm
            file_iter = tqdm(file_paths, desc="Ingesting", unit="file")
        except ImportError:
            file_iter = file_paths

        for file_path in file_iter:
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

        # Create payload indexes for all metadata fields so facet API works
        all_fields = self.vectorstore_wrapper.get_metadata_keys()
        self.vectorstore_wrapper.create_payload_indexes(all_fields)
        self._stored_values_cache = None  # invalidate after ingestion

        logger.info(
            f"Ingestion complete: {stats['processed']}/{stats['total']} documents"
        )
        return stats

    def ingest_directory(
        self,
        directory: str,
        recursive: bool = False,
        extensions: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
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
                "failed": 0, "chunks_created": 0, "errors": [],
            }

        logger.info(f"Found {len(file_paths)} file(s) in {directory}")
        return self.ingest_documents(file_paths)

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

        # Extract metadata once from the full document text (capped at 10k chars in extract())
        # Using chunk_texts[0] (~1000 chars) was too little context to reliably find all fields
        llm_metadata = {}
        if chunk_texts:
            try:
                llm_metadata = self.extract_metadata(text)
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
                "created_at": datetime.now(timezone.utc).isoformat(),
                **llm_metadata,
            }

            documents.append(Document(page_content=chunk_text, metadata=chunk_metadata))

        return documents

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
                self._filter_fields, limit=50
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
                    filters = {
                        k: [i.lower() if isinstance(i, str) else i for i in v] if isinstance(v, list)
                           else v.lower() if isinstance(v, str) else v
                        for k, v in filters.items()
                    }
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
        if filters is None and self._auto_filter:
            filters = self._extract_filters_from_query(query)
        qdrant_filter = self._build_qdrant_filter(filters) if filters else None
        return hybrid_search(self.vectorstore, query, k=k, filters=qdrant_filter)

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
        result = self.vectorstore_wrapper.get_field_values(field_list, limit=limit)
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
