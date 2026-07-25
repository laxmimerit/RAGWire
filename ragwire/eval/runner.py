"""
Running an evaluation and comparing configurations.

The runner turns retrieved documents into the identifier lists the metrics
expect, scores every query, and formats the result as something you can read
in a terminal.

The comparison side exists because a single number is hard to act on. Knowing
recall is 0.71 tells you little; knowing it was 0.58 before you turned
reranking on tells you what to do next.
"""

import logging
from typing import Any, Callable, Dict, List, Optional

from .golden import GoldenSet
from .metrics import mean_metrics, score_query

logger = logging.getLogger(__name__)

METRIC_ORDER = ("recall", "mrr", "hit_rate", "precision")


class QueryResult:
    """
    What happened for one golden query.

    Attributes:
        query: The query that was run
        expected: Identifiers that would have counted as correct
        retrieved: Identifiers actually returned, in rank order. Entries that
            matched nothing appear as ``"<miss>"``.
        metrics: Scores for this query alone
    """

    def __init__(
        self,
        query: str,
        expected: List[str],
        retrieved: List[str],
        metrics: Dict[str, float],
    ):
        self.query = query
        self.expected = expected
        self.retrieved = retrieved
        self.metrics = metrics

    @property
    def missed(self) -> List[str]:
        """Expected documents that never showed up. The list worth reading."""
        return [e for e in self.expected if e not in set(self.retrieved)]

    def __repr__(self) -> str:
        return f"QueryResult(query={self.query!r}, recall={self.metrics['recall']:.2f})"


class EvalResult:
    """
    The outcome of evaluating one configuration against a golden set.

    Attributes:
        label: A name for what was evaluated
        top_k: The cutoff every metric was computed at
        per_query: One QueryResult per golden query
        metrics: The averaged metrics
    """

    def __init__(self, label: str, top_k: int, per_query: List[QueryResult]):
        self.label = label
        self.top_k = top_k
        self.per_query = per_query
        self.metrics = mean_metrics([r.metrics for r in per_query])

    @property
    def failures(self) -> List[QueryResult]:
        """Queries that retrieved nothing correct. Where to start debugging."""
        return [r for r in self.per_query if r.metrics["hit_rate"] == 0.0]

    def to_table(self) -> str:
        """
        Format the aggregate metrics as a plain-text table.

        Returns:
            A table with one row per metric

        Example:
            >>> print(EvalResult("baseline", 5, []).to_table())  # doctest: +SKIP
        """
        lines = [
            f"{self.label}  (top_k={self.top_k}, {len(self.per_query)} queries)",
            "-" * 44,
        ]
        for key in METRIC_ORDER:
            if key in self.metrics:
                lines.append(f"  {key + '@' + str(self.top_k):<16} {self.metrics[key]:.3f}")

        if self.failures:
            lines.append("")
            lines.append(f"  {len(self.failures)} queries retrieved nothing correct:")
            for result in self.failures[:5]:
                lines.append(f"    - {result.query[:60]}")
            if len(self.failures) > 5:
                lines.append(f"    ... and {len(self.failures) - 5} more")

        return "\n".join(lines)

    def __str__(self) -> str:
        return self.to_table()

    def __repr__(self) -> str:
        return f"<EvalResult {self.label!r} recall={self.metrics['recall']:.3f}>"


class SweepResult:
    """
    Several EvalResults compared side by side.

    Attributes:
        results: The individual runs, in the order they were given
    """

    def __init__(self, results: List[EvalResult]):
        self.results = results

    @property
    def best(self) -> Optional[EvalResult]:
        """The variant with the highest recall, or None if nothing ran."""
        if not self.results:
            return None
        return max(self.results, key=lambda r: r.metrics.get("recall", 0.0))

    def to_table(self) -> str:
        """
        Format every variant as one row so the columns line up.

        Returns:
            A comparison table, with each metric shown as a delta against the
            first variant so improvements are readable at a glance.
        """
        if not self.results:
            return "No variants were evaluated."

        width = max(len(r.label) for r in self.results)
        width = max(width, len("variant"))

        # Delta cells are wider than bare numbers, so every column is sized for
        # the widest form to keep the table aligned.
        cell = 11

        header = f"{'variant':<{width}}  " + "  ".join(f"{k:>{cell}}" for k in METRIC_ORDER)
        lines = [header, "-" * len(header)]

        baseline = self.results[0].metrics

        for result in self.results:
            cells = []
            for key in METRIC_ORDER:
                value = result.metrics.get(key, 0.0)
                if result is self.results[0]:
                    cells.append(f"{value:.3f}".rjust(cell))
                else:
                    delta = value - baseline.get(key, 0.0)
                    cells.append(f"{value:.3f}{delta:+.2f}".rjust(cell))
            lines.append(f"{result.label:<{width}}  " + "  ".join(cells))

        winner = self.best
        if winner is not None and len(self.results) > 1:
            lines.append("")
            lines.append(f"Best recall: {winner.label} ({winner.metrics['recall']:.3f})")

        return "\n".join(lines)

    def __str__(self) -> str:
        return self.to_table()


def _label_retrieved(golden: GoldenSet, query_expected: List[str], documents: List[Any]) -> List[str]:
    """
    Map each retrieved document onto the expected identifier it satisfies.

    Scoring by set membership needs both sides to use the same vocabulary, so
    a document that matches an expected entry is labelled with that entry, and
    one that matches nothing is labelled as a miss.
    """
    labels = []
    for document in documents:
        identifier = golden.identify(document)
        hit = next(
            (e for e in query_expected if golden.matches(identifier, e)), None
        )
        labels.append(hit if hit is not None else "<miss>")
    return labels


def evaluate(
    rag: Any,
    golden: GoldenSet,
    top_k: int = 5,
    label: str = "default",
    retrieve: Optional[Callable[..., List[Any]]] = None,
    **retrieve_kwargs: Any,
) -> EvalResult:
    """
    Run every golden query and score what comes back.

    Args:
        rag: A RAGWire instance, or anything with a compatible ``retrieve``
        golden: The golden set to evaluate against
        top_k: How many documents to retrieve and score at
        label: Name for this run, used in output tables
        retrieve: Override the retrieval callable. Mostly useful for testing.
        **retrieve_kwargs: Passed straight through to ``retrieve()``, so
            ``rerank=False`` or any other per-call option works here.

    Returns:
        An EvalResult holding aggregate and per-query scores

    Example:
        >>> from ragwire.eval import GoldenSet, evaluate  # doctest: +SKIP
        >>> golden = GoldenSet.from_file("golden.yaml")  # doctest: +SKIP
        >>> print(evaluate(rag, golden, top_k=5))  # doctest: +SKIP
    """
    retrieve_fn = retrieve or rag.retrieve
    per_query = []

    for entry in golden:
        kwargs = dict(retrieve_kwargs)
        if entry.filters is not None:
            kwargs["filters"] = entry.filters

        try:
            documents = retrieve_fn(entry.query, top_k=top_k, **kwargs)
        except Exception as exc:
            # One broken query should not discard the other nineteen results.
            logger.warning(f"Query failed during evaluation: {entry.query[:60]} ({exc})")
            documents = []

        retrieved = _label_retrieved(golden, entry.expected, documents)
        per_query.append(
            QueryResult(
                query=entry.query,
                expected=entry.expected,
                retrieved=retrieved,
                metrics=score_query(retrieved, entry.expected, k=top_k),
            )
        )

    result = EvalResult(label=label, top_k=top_k, per_query=per_query)
    logger.info(
        f"Evaluated {len(per_query)} queries [{label}]: "
        f"recall={result.metrics['recall']:.3f} mrr={result.metrics['mrr']:.3f}"
    )
    return result


def sweep(
    rag: Any,
    golden: GoldenSet,
    variants: Dict[str, Dict[str, Any]],
    top_k: int = 5,
    retrieve: Optional[Callable[..., List[Any]]] = None,
) -> SweepResult:
    """
    Evaluate several retrieval settings against the same golden set.

    Each variant is a dict of keyword arguments for ``retrieve()``, so this
    answers questions like "did reranking help" or "is top_k=10 better than 5"
    without rebuilding the pipeline between runs.

    Args:
        rag: A RAGWire instance
        golden: The golden set to evaluate against
        variants: Mapping of label to retrieve() kwargs. Insertion order is
            preserved, and the first entry is treated as the baseline that
            later rows are compared against.
        top_k: Default cutoff, overridable per variant via its kwargs
        retrieve: Override the retrieval callable. Mostly useful for testing.

    Returns:
        A SweepResult that formats as a comparison table

    Example:
        >>> print(sweep(rag, golden, {            # doctest: +SKIP
        ...     "no rerank": {"rerank": False},
        ...     "reranked": {"rerank": True},
        ... }))
    """
    results = []
    for label, kwargs in variants.items():
        kwargs = dict(kwargs)
        variant_top_k = kwargs.pop("top_k", top_k)
        results.append(
            evaluate(
                rag,
                golden,
                top_k=variant_top_k,
                label=label,
                retrieve=retrieve,
                **kwargs,
            )
        )
    return SweepResult(results)
