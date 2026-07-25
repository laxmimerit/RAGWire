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

    Example:
        >>> store = QdrantStore(config, embedding)
        >>> store.set_collection("financial_docs")
        >>> vectorstore = store.get_store()
        >>> docs = vectorstore.similarity_search("query", k=5)
    """

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
            logger.info(f"Connected to Qdrant at {url}")

        else:
            # Local file-based storage (path may not exist yet — qdrant creates it)
            self.client = QdrantClient(path=url)
            logger.info(f"Using local Qdrant storage at {url}")

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

    def file_hash_exists(self, file_hash: str) -> bool:
        """
        Check whether a file has already been ingested by its SHA256 hash.

        Args:
            file_hash: SHA256 hash of the file content

        Returns:
            True if at least one chunk with this file_hash exists in the collection
        """
        from qdrant_client.http import models as rest

        if not self.collection_name or not self.collection_exists():
            return False

        results, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=rest.Filter(
                must=[
                    rest.FieldCondition(
                        key="metadata.file_hash",
                        match=rest.MatchValue(value=file_hash),
                    )
                ]
            ),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        return len(results) > 0

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

    def get_field_values(
        self, fields: List[str], limit: int = 50, field_types: Optional[dict] = None
    ) -> dict:
        """
        Return unique values for each requested field using Qdrant's facet API.

        Args:
            fields: List of metadata field names (without the 'metadata.' prefix)
            limit: Max unique values to return per field
            field_types: Optional field name → index type mapping (see
                         create_payload_indexes)

        Returns:
            Dict mapping field name → list of unique values
        """
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
