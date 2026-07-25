"""
Qdrant vector store wrapper for RAG pipeline.

Provides a unified interface for Qdrant vector database operations
including hybrid search (dense + sparse embeddings).
"""

import logging
from typing import Optional, Any, List

logger = logging.getLogger(__name__)


class QdrantStore:
    """
    Qdrant vector store wrapper with hybrid search support.

    Manages connection to Qdrant vector database and provides
    high-level interface for document storage and retrieval.

    Attributes:
        client: QdrantClient instance
        embedding: Embedding model instance
        collection_name: Name of the Qdrant collection
        is_local: True when backed by local file storage rather than a server.
            Local storage silently ignores payload indexes, so field-value
            lookups fall back to scanning points.

    Example:
        >>> store = QdrantStore(config, embedding)
        >>> store.set_collection("financial_docs")
        >>> vectorstore = store.get_store()
        >>> docs = vectorstore.similarity_search("query", k=5)
    """

    #: Overridden per-instance in __init__. Declared here so instances built by
    #: other means (tests, subclasses) still have a sane default.
    is_local = False

    def __init__(
        self, config: dict, embedding: Any, collection_name: Optional[str] = None
    ):
        """
        Initialize Qdrant vector store.

        Args:
            config: Configuration dictionary with Qdrant settings
            embedding: Embedding model instance
            collection_name: Optional collection name to use

        Raises:
            ImportError: If qdrant-client is not installed
            ValueError: If configuration is invalid
        """
        try:
            from qdrant_client import QdrantClient
        except ImportError:
            raise ImportError(
                "qdrant-client is required. Install with: pip install qdrant-client"
            )

        # Extract configuration
        url = config.get("url", "http://localhost:6333")
        api_key = config.get("api_key")

        # Determine connection type
        if url.startswith("http://") or url.startswith("https://"):
            # Remote or local HTTP connection
            self.client = QdrantClient(url=url, api_key=api_key)
            self.is_local = False
            logger.info(f"Connected to Qdrant at {url}")

        else:
            # Local file-based storage (path may not exist yet — qdrant creates it)
            try:
                self.client = QdrantClient(path=url)
            except Exception as e:
                if "already accessed" in str(e).lower() or "lock" in str(e).lower():
                    raise RuntimeError(
                        f"Local Qdrant storage at '{url}' is already in use by "
                        "another process.\n"
                        "Local storage allows exactly one reader/writer, so this "
                        "happens with multi-worker servers (gunicorn/uvicorn "
                        "--workers 2), a second script, or a notebook still "
                        "holding the directory.\n"
                        "Either close the other process, or run a Qdrant server "
                        "and point vectorstore.url at it:\n"
                        "  docker run -p 6333:6333 qdrant/qdrant\n"
                        "  vectorstore:\n"
                        "    url: http://localhost:6333"
                    ) from e
                raise
            self.is_local = True
            logger.info(f"Using local Qdrant storage at {url}")
            logger.info(
                "Local storage does not support payload indexes, so metadata "
                "filter values are collected by scanning points instead of "
                "Qdrant's facet API. This is exact but slower on large "
                "collections — run a server for production workloads: "
                "docker run -p 6333:6333 qdrant/qdrant"
            )

        self.embedding = embedding
        self.collection_name = collection_name
        self.config = config

    def set_collection(self, name: str) -> None:
        """
        Set the collection name for operations.

        Args:
            name: Collection name to use
        """
        self.collection_name = name
        logger.info(f"Collection set to: {name}")

    def get_store(self, use_sparse: bool = False) -> Any:
        """
        Get the LangChain QdrantVectorStore instance.

        Args:
            use_sparse: Whether to enable hybrid search with sparse vectors

        Returns:
            QdrantVectorStore instance configured with current settings

        Raises:
            ValueError: If collection_name is not set
        """
        if not self.collection_name:
            raise ValueError("Collection name not set. Call set_collection() first.")

        try:
            from langchain_qdrant import QdrantVectorStore
        except ImportError:
            raise ImportError(
                "langchain-qdrant is required. Install with: pip install langchain-qdrant"
            )

        if use_sparse:
            try:
                from langchain_qdrant import RetrievalMode, FastEmbedSparse

                return QdrantVectorStore(
                    client=self.client,
                    collection_name=self.collection_name,
                    embedding=self.embedding,
                    sparse_embedding=FastEmbedSparse(),
                    retrieval_mode=RetrievalMode.HYBRID,
                )
            except ImportError:
                logger.warning(
                    "FastEmbedSparse not available. Falling back to dense search. "
                    "Install with: pip install fastembed"
                )

        return QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embedding,
        )

    def create_collection(
        self, collection_name: Optional[str] = None, use_sparse: bool = True
    ) -> None:
        """
        Create a new Qdrant collection.

        Args:
            collection_name: Name of collection (uses current if not provided)
            use_sparse: Whether to enable sparse vectors for hybrid search
        """
        name = collection_name or self.collection_name

        if not name:
            raise ValueError("Collection name must be provided")

        try:
            from qdrant_client.http import models as rest
        except ImportError:
            raise ImportError(
                "qdrant-client is required. Install with: pip install qdrant-client"
            )

        # Get embedding dimension
        test_embedding = self.embedding.embed_query("test")
        vector_size = len(test_embedding)

        # Configure vector schema
        if use_sparse:
            # Hybrid search with dense and sparse vectors
            vectors_config = rest.VectorParams(
                size=vector_size, distance=rest.Distance.COSINE
            )

            self.client.create_collection(
                collection_name=name,
                vectors_config=vectors_config,
                sparse_vectors_config={
                    "langchain-sparse": rest.SparseVectorParams(index=rest.SparseIndexParams())
                },
            )
            logger.info(f"Created collection '{name}' with hybrid search")
        else:
            # Dense vectors only
            self.client.create_collection(
                collection_name=name,
                vectors_config=rest.VectorParams(
                    size=vector_size, distance=rest.Distance.COSINE
                ),
            )
            logger.info(f"Created collection '{name}' with dense vectors only")

    def delete_collection(self, collection_name: Optional[str] = None) -> None:
        """
        Delete a Qdrant collection.

        Args:
            collection_name: Name of collection to delete
        """
        name = collection_name or self.collection_name

        if not name:
            raise ValueError("Collection name must be provided")

        self.client.delete_collection(name)
        logger.info(f"Deleted collection: {name}")

    def collection_exists(self, collection_name: Optional[str] = None) -> bool:
        """
        Check if a collection exists.

        Args:
            collection_name: Name of collection to check

        Returns:
            True if collection exists, False otherwise
        """
        name = collection_name or self.collection_name

        if not name:
            return False

        collections = self.client.get_collections().collections
        return any(col.name == name for col in collections)

    def _file_hash_filter(self, file_hash: str) -> Any:
        """Build a Qdrant filter matching every chunk of one source file."""
        from qdrant_client.http import models as rest

        return rest.Filter(
            must=[
                rest.FieldCondition(
                    key="metadata.file_hash",
                    match=rest.MatchValue(value=file_hash),
                )
            ]
        )

    def file_hash_exists(self, file_hash: str) -> bool:
        """
        Check whether any chunk of a file is present, by its SHA256 hash.

        Note: presence is not the same as a complete ingest — a run that failed
        partway leaves chunks behind. Use get_ingest_status() to distinguish.

        Args:
            file_hash: SHA256 hash of the file content

        Returns:
            True if at least one chunk with this file_hash exists in the collection
        """
        return self.count_by_file_hash(file_hash) > 0

    def count_by_file_hash(self, file_hash: str) -> int:
        """
        Count how many chunks of a given file are stored in the collection.

        Args:
            file_hash: SHA256 hash of the file content

        Returns:
            Number of stored chunks (0 if the collection or file is absent)
        """
        if not self.collection_name or not self.collection_exists():
            return 0

        result = self.client.count(
            collection_name=self.collection_name,
            count_filter=self._file_hash_filter(file_hash),
            exact=True,
        )
        return result.count

    def get_ingest_status(self, file_hash: str) -> tuple:
        """
        Determine whether a file is fully ingested, partially ingested, or absent.

        Every chunk records the document's ``total_chunks``, so a complete ingest
        is one where the stored chunk count matches that number. A run that died
        partway through ``add_documents`` leaves fewer — without this check the
        leftover chunks make the file look already-ingested and it is skipped
        forever, leaving the document permanently truncated.

        Args:
            file_hash: SHA256 hash of the file content

        Returns:
            Tuple of (status, stored_count, expected_count) where status is
            "absent", "partial", or "complete". expected_count is None when
            nothing is stored.
        """
        stored = self.count_by_file_hash(file_hash)
        if stored == 0:
            return ("absent", 0, None)

        results, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=self._file_hash_filter(file_hash),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )

        expected = None
        if results:
            metadata = (results[0].payload or {}).get("metadata", {})
            if isinstance(metadata, dict):
                expected = metadata.get("total_chunks")

        if not isinstance(expected, int) or expected <= 0:
            # No usable marker — treat presence as complete rather than
            # re-ingesting data written by an older RAGWire version.
            return ("complete", stored, None)

        return ("complete" if stored >= expected else "partial", stored, expected)

    def delete_by_file_hash(self, file_hash: str) -> int:
        """
        Delete every stored chunk belonging to one source file.

        Used to clear a partial ingest before retrying, and to remove a document
        that is being replaced.

        Args:
            file_hash: SHA256 hash of the file content

        Returns:
            Number of chunks that were present before deletion
        """
        count = self.count_by_file_hash(file_hash)
        if count == 0:
            return 0

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=self._file_hash_filter(file_hash),
            wait=True,
        )
        logger.info(f"Deleted {count} chunk(s) for file_hash {file_hash[:12]}…")
        return count

    def delete_by_source(
        self, source: str, except_file_hash: Optional[str] = None
    ) -> int:
        """
        Delete stored chunks by their source path, optionally sparing one version.

        Deduplication is keyed on file content, so an edited document hashes
        differently and would otherwise be stored *alongside* its previous
        version — leaving the old text retrievable forever. Passing the new hash
        as ``except_file_hash`` removes only the stale copies.

        Args:
            source: Source path recorded in chunk metadata
            except_file_hash: File hash to preserve (the version being written)

        Returns:
            Number of chunks deleted
        """
        from qdrant_client.http import models as rest

        if not self.collection_name or not self.collection_exists():
            return 0

        must = [
            rest.FieldCondition(
                key="metadata.source", match=rest.MatchValue(value=source)
            )
        ]
        must_not = []
        if except_file_hash:
            must_not.append(
                rest.FieldCondition(
                    key="metadata.file_hash",
                    match=rest.MatchValue(value=except_file_hash),
                )
            )

        selector = rest.Filter(must=must, must_not=must_not or None)

        count = self.client.count(
            collection_name=self.collection_name,
            count_filter=selector,
            exact=True,
        ).count
        if count == 0:
            return 0

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=selector,
            wait=True,
        )
        logger.info(f"Deleted {count} stale chunk(s) for source: {source}")
        return count

    def get_collection_info(self, collection_name: Optional[str] = None) -> dict:
        """
        Get information about a collection.

        Args:
            collection_name: Name of collection

        Returns:
            Dictionary with collection information
        """
        name = collection_name or self.collection_name

        if not name:
            raise ValueError("Collection name must be provided")

        return self.client.get_collection(name)

    def get_vector_size(self, collection_name: Optional[str] = None) -> Optional[int]:
        """
        Return the dense vector dimension an existing collection was created with.

        Args:
            collection_name: Name of collection (uses current if not provided)

        Returns:
            Vector dimension, or None if it cannot be determined
        """
        try:
            info = self.get_collection_info(collection_name)
            vectors = info.config.params.vectors
        except Exception as e:
            logger.debug(f"Could not read vector size: {e}")
            return None

        if hasattr(vectors, "size"):
            return vectors.size
        if isinstance(vectors, dict) and vectors:
            # Named vectors — RAGWire writes a single unnamed dense vector, but
            # read the first entry so externally-created collections still work.
            return next(iter(vectors.values())).size
        return None

    #: How many points to sample when discovering which metadata keys exist.
    #: One point is not enough — a field that was None for that document is
    #: absent from its payload and would never get an index.
    _KEY_DISCOVERY_SAMPLE = 100

    def get_metadata_keys(self, sample_size: Optional[int] = None) -> List[str]:
        """
        Return the union of metadata payload keys across a sample of points.

        Sampling several points matters: any field that happened to be null for
        the sampled document is missing from its payload, so a single-point
        sample silently under-reports the schema.

        Args:
            sample_size: Points to sample (defaults to _KEY_DISCOVERY_SAMPLE)

        Returns:
            Sorted list of metadata field names, or empty list if collection is empty
        """
        if not self.collection_name or not self.collection_exists():
            return []

        results, _ = self.client.scroll(
            collection_name=self.collection_name,
            limit=sample_size or self._KEY_DISCOVERY_SAMPLE,
            with_payload=True,
            with_vectors=False,
        )

        keys = set()
        for point in results:
            metadata = (point.payload or {}).get("metadata", {})
            if isinstance(metadata, dict):
                keys.update(metadata.keys())

        return sorted(keys)

    # Index types for the system fields RAGWire always writes. Schema-defined
    # fields get their type from the metadata schema instead — see
    # MetadataExtractor.field_types.
    _SYSTEM_FIELD_TYPES = {
        "chunk_index": "integer",
        "total_chunks": "integer",
    }

    def create_payload_indexes(
        self, fields: List[str], field_types: Optional[dict] = None
    ) -> None:
        """
        Create payload indexes for a list of metadata fields.

        Required by Qdrant's facet API. Safe to call multiple times — fields that
        are already indexed are skipped.

        The index type comes from ``field_types`` when supplied (normally derived
        from the metadata schema via MetadataExtractor.field_types). Guessing from
        a hardcoded field-name list breaks custom schemas: an integer field
        indexed as KEYWORD does not index its values, so facets come back empty
        and every filter on it matches zero points.

        Args:
            fields: List of metadata field names (without the 'metadata.' prefix)
            field_types: Optional mapping of field name → "integer" | "float" |
                         "keyword". Unlisted fields default to keyword.
        """
        if self.is_local:
            # Local storage accepts the call, ignores it, and emits a UserWarning
            # for every field. get_field_values() uses a scan instead.
            logger.debug("Local Qdrant — skipping payload indexes (not supported)")
            return

        from qdrant_client.http import models as rest

        schema_by_type = {
            "integer": rest.PayloadSchemaType.INTEGER,
            "float": rest.PayloadSchemaType.FLOAT,
            "keyword": rest.PayloadSchemaType.KEYWORD,
        }

        resolved = {**self._SYSTEM_FIELD_TYPES, **(field_types or {})}

        for field in fields:
            schema = schema_by_type.get(
                resolved.get(field, "keyword"), rest.PayloadSchemaType.KEYWORD
            )
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=f"metadata.{field}",
                    field_schema=schema,
                )
                logger.debug(f"Payload index created for field '{field}' (type={schema})")
            except Exception as e:
                if self._is_already_exists(e):
                    logger.debug(f"Payload index for '{field}' already exists")
                else:
                    # Auth failures, connection resets and schema-type conflicts
                    # all land here. Swallowing them silently makes empty facet
                    # results impossible to diagnose.
                    logger.warning(
                        f"Could not create payload index for '{field}' "
                        f"(type={resolved.get(field, 'keyword')}): {e}"
                    )

    @staticmethod
    def _is_already_exists(error: Exception) -> bool:
        """Return True if a Qdrant error means the index is already present."""
        message = str(error).lower()
        return "already exists" in message or "already indexed" in message

    #: Points scanned by the local-storage fallback before giving up. Bounds the
    #: cost on large collections; local storage is a development mode anyway.
    _SCAN_LIMIT = 10000

    def get_field_values(
        self, fields: List[str], limit: int = 50, field_types: Optional[dict] = None
    ) -> dict:
        """
        Return unique values for each requested field.

        Uses Qdrant's facet API against a server, which is exact and fast at any
        collection size. Local file storage silently ignores payload indexes, and
        the facet API needs them — there, values are collected by scanning points
        instead, so metadata filtering still works out of the box.

        Args:
            fields: List of metadata field names (without the 'metadata.' prefix)
            limit: Max unique values to return per field
            field_types: Optional field name → index type mapping (see
                         create_payload_indexes)

        Returns:
            Dict mapping field name → list of unique values
        """
        if not self.collection_name or not self.collection_exists():
            return {field: [] for field in fields}

        if self.is_local:
            return self._scan_field_values(fields, limit=limit)

        self.create_payload_indexes(fields, field_types=field_types)
        result = {}
        for field in fields:
            try:
                facet_result = self.client.facet(
                    collection_name=self.collection_name,
                    key=f"metadata.{field}",
                    limit=limit,
                )
                result[field] = [hit.value for hit in facet_result.hits]
            except Exception as e:
                logger.warning(f"Could not get values for field '{field}': {e}")
                result[field] = []

        return result

    def _scan_field_values(
        self, fields: List[str], limit: int = 50, max_points: Optional[int] = None
    ) -> dict:
        """
        Collect unique field values by scrolling through stored points.

        Fallback for local storage, where the facet API is unavailable. Stops
        early once every requested field has `limit` values, so the common case
        (a handful of companies or doc types) reads only the first page.

        Args:
            fields: Metadata field names (without the 'metadata.' prefix)
            limit: Max unique values to collect per field
            max_points: Cap on points scanned (defaults to _SCAN_LIMIT)

        Returns:
            Dict mapping field name → list of unique values, in first-seen order
        """
        budget = max_points or self._SCAN_LIMIT
        values: dict = {field: [] for field in fields}
        seen: dict = {field: set() for field in fields}
        offset = None
        scanned = 0

        while scanned < budget:
            batch, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=min(256, budget - scanned),
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            if not batch:
                break
            scanned += len(batch)

            for point in batch:
                metadata = (point.payload or {}).get("metadata", {})
                if not isinstance(metadata, dict):
                    continue
                for field in fields:
                    if len(values[field]) >= limit:
                        continue
                    raw = metadata.get(field)
                    if raw is None:
                        continue
                    for item in (raw if isinstance(raw, list) else [raw]):
                        if not isinstance(item, (str, int, float, bool)):
                            continue
                        if item not in seen[field]:
                            seen[field].add(item)
                            values[field].append(item)

            if all(len(values[f]) >= limit for f in fields) or offset is None:
                break

        if scanned >= budget:
            logger.warning(
                f"Field-value scan stopped at {budget} points — values may be "
                "incomplete. Use a Qdrant server for exact results at this size."
            )
        logger.debug(f"Scanned {scanned} point(s) for field values")
        return values
