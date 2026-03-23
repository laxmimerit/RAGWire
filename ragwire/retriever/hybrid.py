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
    vectorstore: Any, top_k: int = 5, search_type: str = "similarity",
    fetch_k: int = 20, lambda_mult: float = 0.5,
) -> Any:
    """
    Create a retriever from a vector store with configured parameters.

    Args:
        vectorstore: QdrantVectorStore instance
        top_k: Number of documents to retrieve
        search_type: Search type ("similarity", "mmr", "hybrid")
        fetch_k: Candidates fetched before MMR selection (mmr only)
        lambda_mult: Diversity parameter for MMR (mmr only)

    Returns:
        Configured retriever instance

    Example:
        >>> retriever = get_retriever(vectorstore, top_k=10, search_type="mmr")
        >>> docs = retriever.invoke("What is Amazon's revenue?")
    """
    if search_type == "mmr":
        return vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": top_k, "fetch_k": fetch_k, "lambda_mult": lambda_mult},
        )

    # Both "similarity" and "hybrid" use the same retriever setup —
    # hybrid mode is configured at the vectorstore level (RetrievalMode.HYBRID)
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )


def hybrid_search(
    vectorstore: Any,
    query: str,
    k: int = 5,
    filters: Optional[Dict[str, Any]] = None,
) -> List[Any]:
    """
    Perform hybrid search on the vector store.

    Args:
        vectorstore: QdrantVectorStore instance
        query: Search query string
        k: Number of results to return
        filters: Optional metadata filters

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
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )

    if filters:
        retriever.search_kwargs["filter"] = filters

    return retriever.invoke(query)


def mmr_search(
    vectorstore: Any,
    query: str,
    k: int = 5,
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
    filters: Optional[Dict[str, Any]] = None,
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
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult},
    )

    if filters:
        retriever.search_kwargs["filter"] = filters

    return retriever.invoke(query)
