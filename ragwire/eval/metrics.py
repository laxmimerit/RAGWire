"""
Retrieval metrics.

These functions work on plain lists of identifiers so they can be tested and
reasoned about without a vector store, an LLM or a network. The runner is what
turns retrieved documents into the identifier lists these expect.

Every metric takes ``retrieved`` in rank order (best first) and ``expected`` as
the set of identifiers that would count as a correct hit.
"""

from typing import Dict, Iterable, List, Optional, Sequence


def recall_at_k(retrieved: Sequence[str], expected: Iterable[str], k: Optional[int] = None) -> float:
    """
    Fraction of the expected documents that appear in the top k.

    This is the metric that matters most for RAG. A chunk that was never
    retrieved cannot be cited, so recall puts a ceiling on how good any
    downstream answer can be.

    Args:
        retrieved: Retrieved identifiers in rank order
        expected: Identifiers that count as correct
        k: Cutoff. Uses the whole list if not given.

    Returns:
        A value from 0.0 to 1.0. Returns 0.0 when nothing is expected, since
        a query with no correct answer cannot be scored.

    Example:
        >>> recall_at_k(["a", "b", "c"], ["a", "d"], k=3)
        0.5
    """
    expected_set = set(expected)
    if not expected_set:
        return 0.0

    window = list(retrieved)[:k] if k is not None else list(retrieved)
    found = expected_set.intersection(window)
    return len(found) / len(expected_set)


def precision_at_k(retrieved: Sequence[str], expected: Iterable[str], k: Optional[int] = None) -> float:
    """
    Fraction of the top k results that are correct.

    Low precision means the LLM is being handed irrelevant chunks alongside
    the useful ones, which costs tokens and invites distraction.

    Args:
        retrieved: Retrieved identifiers in rank order
        expected: Identifiers that count as correct
        k: Cutoff. Uses the whole list if not given.

    Returns:
        A value from 0.0 to 1.0. Returns 0.0 when nothing was retrieved.

    Example:
        >>> precision_at_k(["a", "b", "c", "d"], ["a", "c"], k=4)
        0.5
    """
    window = list(retrieved)[:k] if k is not None else list(retrieved)
    if not window:
        return 0.0

    expected_set = set(expected)
    hits = sum(1 for item in window if item in expected_set)
    return hits / len(window)


def hit_rate_at_k(retrieved: Sequence[str], expected: Iterable[str], k: Optional[int] = None) -> float:
    """
    1.0 if any correct document made the top k, otherwise 0.0.

    The bluntest useful measure: did retrieval find anything at all? A hit
    rate well below 1.0 means some queries are unanswerable no matter how
    good the generation step is.

    Example:
        >>> hit_rate_at_k(["a", "b"], ["b"], k=2)
        1.0
        >>> hit_rate_at_k(["a", "b"], ["z"], k=2)
        0.0
    """
    window = list(retrieved)[:k] if k is not None else list(retrieved)
    expected_set = set(expected)
    return 1.0 if any(item in expected_set for item in window) else 0.0


def reciprocal_rank(retrieved: Sequence[str], expected: Iterable[str], k: Optional[int] = None) -> float:
    """
    1 / rank of the first correct result, or 0.0 if there is none.

    Unlike recall, this is sensitive to ordering, which is exactly what a
    reranker changes. If reranking helps but recall stays flat, MRR is where
    the improvement shows up.

    Example:
        >>> reciprocal_rank(["wrong", "right"], ["right"])
        0.5
        >>> reciprocal_rank(["right", "wrong"], ["right"])
        1.0
    """
    window = list(retrieved)[:k] if k is not None else list(retrieved)
    expected_set = set(expected)

    for position, item in enumerate(window, start=1):
        if item in expected_set:
            return 1.0 / position
    return 0.0


def score_query(
    retrieved: Sequence[str], expected: Iterable[str], k: Optional[int] = None
) -> Dict[str, float]:
    """
    Compute every metric for one query.

    Args:
        retrieved: Retrieved identifiers in rank order
        expected: Identifiers that count as correct
        k: Cutoff applied to all metrics

    Returns:
        A dict with recall, precision, hit_rate and mrr

    Example:
        >>> sorted(score_query(["a"], ["a"]).items())
        [('hit_rate', 1.0), ('mrr', 1.0), ('precision', 1.0), ('recall', 1.0)]
    """
    expected_list = list(expected)
    return {
        "recall": recall_at_k(retrieved, expected_list, k),
        "precision": precision_at_k(retrieved, expected_list, k),
        "hit_rate": hit_rate_at_k(retrieved, expected_list, k),
        "mrr": reciprocal_rank(retrieved, expected_list, k),
    }


def mean_metrics(per_query: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Average each metric across queries.

    Args:
        per_query: One metric dict per query, as returned by score_query

    Returns:
        A dict with the same keys, averaged. Empty input gives 0.0 for every
        known metric rather than an empty dict, so callers can format results
        without special-casing.

    Example:
        >>> mean_metrics([{"recall": 1.0}, {"recall": 0.0}])
        {'recall': 0.5}
    """
    if not per_query:
        return {"recall": 0.0, "precision": 0.0, "hit_rate": 0.0, "mrr": 0.0}

    keys = per_query[0].keys()
    return {
        key: sum(row.get(key, 0.0) for row in per_query) / len(per_query)
        for key in keys
    }
