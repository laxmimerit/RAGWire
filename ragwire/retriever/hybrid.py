"""
Hybrid retriever for Qdrant vector store.

Provides advanced retrieval strategies including:
- Dense search (semantic similarity)
- Sparse search (keyword matching via BM25)
- Hybrid search (combined dense + sparse)
- Maximal Marginal Relevance (MMR) for diversity
"""

import logging
from typing import Optional, Any, List, Dict

logger = logging.getLogger(__name__)


def get_retriever(
    vectorstore: Any, top_k: int = 5, search_type: str = "similarity", **kwargs
) -> Any:
    """
    Create a retriever from a vector store with configured parameters.

    Args:
        vectorstore: QdrantVectorStore instance
        top_k: Number of documents to retrieve
        search_type: Search type ("similarity", "mmr", "hybrid")
        **kwargs: Additional search parameters

    Returns:
        Configured retriever instance

    Example:
        >>> retriever = get_retriever(vectorstore, top_k=10, search_type="mmr")
        >>> docs = retriever.invoke("What is Amazon's revenue?")
    """
    if search_type == "mmr":
        # Maximal Marginal Relevance for diverse results
        return vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": top_k,
                "fetch_k": kwargs.get("fetch_k", 20),
                "lambda_mult": kwargs.get("lambda_mult", 0.5),
            },
        )

    elif search_type == "hybrid":
        # Hybrid search (dense + sparse)
        return get_hybrid_retriever(vectorstore, top_k, **kwargs)

    else:
        # Default similarity search
        return vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": top_k}
        )


def get_hybrid_retriever(
    vectorstore: Any,
    top_k: int = 5,
    **kwargs,
) -> Any:
    """
    Create a hybrid retriever combining dense and sparse search.

    Hybrid search is configured at the QdrantVectorStore level (via
    RetrievalMode.HYBRID). This function simply wraps the vectorstore
    as a retriever — the vectorstore must already be initialized with
    hybrid mode via QdrantStore.get_store(use_sparse=True).

    Args:
        vectorstore: QdrantVectorStore instance (with hybrid mode enabled)
        top_k: Number of documents to retrieve
        **kwargs: Additional search parameters

    Returns:
        Configured retriever

    Note:
        Requires Qdrant collection to be created with use_sparse=True
        and the vectorstore initialized with use_sparse=True.
    """
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k, **kwargs},
    )


def hybrid_search(
    vectorstore: Any,
    query: str,
    k: int = 5,
    filters: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> List[Any]:
    """
    Perform hybrid search on the vector store.

    Args:
        vectorstore: QdrantVectorStore instance
        query: Search query string
        k: Number of results to return
        filters: Optional metadata filters
        **kwargs: Additional search parameters

    Returns:
        List of retrieved documents

    Example:
        >>> results = hybrid_search(
        ...     vectorstore,
        ...     "Amazon Q1 2024 revenue",
        ...     k=10,
        ...     filters={"company_name": "amazon"}
        ... )
    """
    retriever = get_hybrid_retriever(vectorstore, k, **kwargs)

    if filters:
        # Apply metadata filters
        retriever.search_kwargs["filter"] = filters

    return retriever.invoke(query)


def mmr_search(
    vectorstore: Any,
    query: str,
    k: int = 5,
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
    filters: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> List[Any]:
    """
    Perform MMR (Maximal Marginal Relevance) search.

    MMR provides diverse results by balancing relevance with
    diversity, avoiding similar duplicate results.

    Args:
        vectorstore: QdrantVectorStore instance
        query: Search query string
        k: Number of results to return
        fetch_k: Number of results to fetch before MMR selection
        lambda_mult: Diversity parameter (0=least diverse, 1=most diverse)
        filters: Optional metadata filters
        **kwargs: Additional search parameters

    Returns:
        List of retrieved documents

    Example:
        >>> results = mmr_search(
        ...     vectorstore,
        ...     "tech company earnings",
        ...     k=10,
        ...     lambda_mult=0.7  # More diverse results
        ... )
    """
    retriever = get_retriever(
        vectorstore,
        top_k=k,
        search_type="mmr",
        fetch_k=fetch_k,
        lambda_mult=lambda_mult,
    )

    if filters:
        retriever.search_kwargs["filter"] = filters

    return retriever.invoke(query)
