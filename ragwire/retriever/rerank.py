"""
Rerankers for second-stage retrieval.

First-stage retrieval (dense, sparse or hybrid) scores the query and the
document separately, then compares the two vectors. A reranker instead reads
the query and each candidate together and scores the pair directly, which is
far more accurate but far too slow to run over a whole collection. The usual
arrangement, and the one RAGWire uses, is to retrieve a wide candidate pool
cheaply and rerank it down to the handful of chunks you actually keep.

Two providers ship with the package:

- ``cross_encoder`` runs a local sentence-transformers model. No API key, no
  network calls after the first download, and it is the default.
- ``cohere`` calls the hosted Cohere Rerank endpoint. Needs ``COHERE_API_KEY``.

Neither dependency is installed by default. Install the one you want with
``pip install ragwire[rerank]`` or ``pip install ragwire[cohere]``.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_CROSS_ENCODER_MODEL = "BAAI/bge-reranker-base"
DEFAULT_COHERE_MODEL = "rerank-v3.5"


class BaseReranker:
    """
    Common behaviour for rerankers.

    Subclasses implement :meth:`_score`, which returns one relevance score per
    document in the order it was given. Ordering, truncation and score
    attachment are handled here so every provider behaves identically.
    """

    name = "base"

    def rerank(
        self, query: str, documents: List[Any], top_n: Optional[int] = None
    ) -> List[Any]:
        """
        Reorder documents by relevance to the query.

        Args:
            query: The search query the documents were retrieved for
            documents: Candidate documents from first-stage retrieval
            top_n: How many documents to keep. Keeps all of them if not given.

        Returns:
            Documents sorted by descending relevance, truncated to top_n. Each
            returned document carries its score in ``metadata["rerank_score"]``.
        """
        if not documents:
            return []

        # A single candidate is already in its final order, and scoring it
        # would cost a model call that cannot change the outcome.
        if len(documents) == 1:
            return documents if top_n is None or top_n >= 1 else []

        scores = self._score(query, [d.page_content for d in documents])

        if len(scores) != len(documents):
            raise ValueError(
                f"{self.name} reranker returned {len(scores)} scores for "
                f"{len(documents)} documents"
            )

        ranked = sorted(zip(documents, scores), key=lambda pair: pair[1], reverse=True)
        if top_n is not None:
            ranked = ranked[:top_n]

        for doc, score in ranked:
            # Documents come back from the vector store freshly constructed on
            # every query, so annotating metadata here does not leak into the
            # stored payload.
            doc.metadata["rerank_score"] = float(score)

        return [doc for doc, _ in ranked]

    def _score(self, query: str, texts: List[str]) -> List[float]:
        raise NotImplementedError


class CrossEncoderReranker(BaseReranker):
    """
    Local cross-encoder reranker backed by sentence-transformers.

    The model is downloaded on first use rather than at construction, so
    building a RAGWire instance for ingestion never pays for a model it will
    not use. The package itself is checked eagerly, so a missing dependency
    surfaces at startup instead of on the first query.
    """

    name = "cross_encoder"

    def __init__(
        self,
        model: str = DEFAULT_CROSS_ENCODER_MODEL,
        batch_size: int = 32,
        device: Optional[str] = None,
    ):
        try:
            import sentence_transformers  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "The cross_encoder reranker requires sentence-transformers. "
                "Install it with: pip install ragwire[rerank]"
            ) from exc

        self.model_name = model
        self.batch_size = batch_size
        self.device = device
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info(f"Loading cross-encoder reranker: {self.model_name}")
            kwargs: Dict[str, Any] = {}
            if self.device:
                kwargs["device"] = self.device
            self._model = CrossEncoder(self.model_name, **kwargs)
        return self._model

    def _score(self, query: str, texts: List[str]) -> List[float]:
        model = self._load()
        pairs = [(query, text) for text in texts]
        scores = model.predict(pairs, batch_size=self.batch_size)
        return [float(s) for s in scores]


class CohereReranker(BaseReranker):
    """
    Hosted reranker backed by the Cohere Rerank endpoint.

    Cohere returns only the documents it ranked, so scores are mapped back onto
    the original positions before sorting. Any document Cohere omits keeps a
    score low enough to sort last rather than being silently dropped, which
    keeps the contract of returning every input document intact.
    """

    name = "cohere"

    def __init__(
        self,
        model: str = DEFAULT_COHERE_MODEL,
        api_key: Optional[str] = None,
    ):
        try:
            import cohere
        except ImportError as exc:
            raise ImportError(
                "The cohere reranker requires the cohere SDK. "
                "Install it with: pip install ragwire[cohere]"
            ) from exc

        import os

        key = api_key or os.getenv("COHERE_API_KEY")
        if not key:
            raise ValueError(
                "The cohere reranker needs an API key. Set COHERE_API_KEY in "
                "your environment or .env file, or pass api_key in the config."
            )

        self.model_name = model
        self._client = cohere.ClientV2(api_key=key)

    def _score(self, query: str, texts: List[str]) -> List[float]:
        response = self._client.rerank(
            model=self.model_name,
            query=query,
            documents=texts,
            top_n=len(texts),
        )

        scores = [float("-inf")] * len(texts)
        for result in response.results:
            scores[result.index] = float(result.relevance_score)
        return scores


PROVIDERS = {
    "cross_encoder": CrossEncoderReranker,
    "cohere": CohereReranker,
}


def get_reranker(config: Optional[Dict[str, Any]]) -> Optional[BaseReranker]:
    """
    Build a reranker from a ``retriever.rerank`` config block.

    Returns None when reranking is not configured or is explicitly disabled,
    which is what keeps this feature free for everyone who does not use it.

    Args:
        config: The ``retriever.rerank`` mapping, or None

    Returns:
        A reranker instance, or None when reranking is off

    Raises:
        ValueError: If the provider is unknown

    Example:
        >>> get_reranker({"provider": "cross_encoder"})  # doctest: +SKIP
        <CrossEncoderReranker ...>
        >>> get_reranker(None) is None
        True
    """
    if not config:
        return None

    # Presence of the block is enough to turn reranking on. "enabled: false"
    # exists so a config can keep its tuned settings while switching it off.
    if not config.get("enabled", True):
        return None

    provider = config.get("provider", "cross_encoder")
    if provider not in PROVIDERS:
        raise ValueError(
            f"Unknown rerank provider: '{provider}'. "
            f"Available: {', '.join(sorted(PROVIDERS))}"
        )

    kwargs = {k: v for k, v in config.items() if k not in ("enabled", "provider", "fetch_k")}
    return PROVIDERS[provider](**kwargs)


def resolve_fetch_k(config: Optional[Dict[str, Any]], top_k: int) -> int:
    """
    Decide how many candidates first-stage retrieval should return.

    Reranking can only reorder what it is given, so the candidate pool has to
    be wider than the final result set for it to have anything to do. The
    default of four times top_k is wide enough to matter and small enough that
    a local cross-encoder stays fast.

    Args:
        config: The ``retriever.rerank`` mapping, or None
        top_k: The number of documents the caller ultimately wants

    Returns:
        The candidate count to request from the vector store

    Example:
        >>> resolve_fetch_k({"provider": "cohere"}, top_k=5)
        20
        >>> resolve_fetch_k({"fetch_k": 50}, top_k=5)
        50
    """
    if not config:
        return top_k

    fetch_k = config.get("fetch_k")
    if fetch_k is None:
        fetch_k = max(4 * top_k, 20)

    # Fetching fewer candidates than the caller asked to keep would make the
    # reranker silently shrink the result set.
    return max(int(fetch_k), top_k)
