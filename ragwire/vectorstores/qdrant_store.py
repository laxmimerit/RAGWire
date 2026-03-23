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

    def get_metadata_keys(self) -> List[str]:
        """
        Scroll one point and return all metadata payload keys present in the collection.

        Returns:
            List of metadata field names, or empty list if collection is empty
        """
        results, _ = self.client.scroll(
            collection_name=self.collection_name,
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        if not results:
            return []
        payload = results[0].payload or {}
        metadata = payload.get("metadata", {})
        return list(metadata.keys())

    def create_payload_indexes(self, fields: List[str]) -> None:
        """
        Create keyword payload indexes for a list of metadata fields.

        Required by Qdrant's facet API. Safe to call multiple times —
        silently skips fields that are already indexed.

        Args:
            fields: List of metadata field names (without the 'metadata.' prefix)
        """
        from qdrant_client.http import models as rest

        for field in fields:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=f"metadata.{field}",
                    field_schema=rest.PayloadSchemaType.KEYWORD,
                )
                logger.debug(f"Payload index created for field: {field}")
            except Exception:
                pass  # Already exists — safe to ignore

    def get_field_values(self, fields: List[str], limit: int = 50) -> dict:
        """
        Return unique values for each requested field using Qdrant's facet API.

        Args:
            fields: List of metadata field names (without the 'metadata.' prefix)
            limit: Max unique values to return per field

        Returns:
            Dict mapping field name → list of unique values
        """
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
