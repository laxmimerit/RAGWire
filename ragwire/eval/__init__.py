"""
Retrieval evaluation for RAGWire.

Tuning retrieval without measuring it is guesswork. This module gives you the
numbers: point it at a golden set of queries with known answers and it reports
recall, MRR, hit rate and precision, or compares several configurations side
by side.

Nothing here needs an extra dependency. It reads golden sets with the YAML
parser the package already requires, and everything else is arithmetic.

Example:
    >>> from ragwire import RAGWire
    >>> from ragwire.eval import GoldenSet, evaluate, sweep
    >>>
    >>> rag = RAGWire("config.yaml")                     # doctest: +SKIP
    >>> golden = GoldenSet.from_file("golden.yaml")      # doctest: +SKIP
    >>> print(evaluate(rag, golden, top_k=5))            # doctest: +SKIP
    >>>
    >>> print(sweep(rag, golden, {                       # doctest: +SKIP
    ...     "baseline": {"rerank": False},
    ...     "reranked": {"rerank": True},
    ... }))
"""

from .golden import GoldenQuery, GoldenSet
from .metrics import (
    hit_rate_at_k,
    mean_metrics,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    score_query,
)
from .runner import EvalResult, QueryResult, SweepResult, evaluate, sweep

__all__ = [
    # Golden sets
    "GoldenQuery",
    "GoldenSet",
    # Metrics
    "recall_at_k",
    "precision_at_k",
    "hit_rate_at_k",
    "reciprocal_rank",
    "score_query",
    "mean_metrics",
    # Running
    "evaluate",
    "sweep",
    "EvalResult",
    "QueryResult",
    "SweepResult",
]
