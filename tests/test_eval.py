"""
Tests for the evaluation module.

Metrics are pure arithmetic and tested directly. The runner is tested against
a stub retriever, so no vector store, LLM or network is involved.
"""

import json

import pytest
from langchain_core.documents import Document

from ragwire.eval import (
    EvalResult,
    GoldenQuery,
    GoldenSet,
    evaluate,
    hit_rate_at_k,
    mean_metrics,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    score_query,
    sweep,
)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def test_recall_counts_expected_documents_found():
    assert recall_at_k(["a", "b", "c"], ["a", "d"], k=3) == 0.5
    assert recall_at_k(["a", "d"], ["a", "d"], k=3) == 1.0


def test_recall_respects_the_cutoff():
    # The correct document is at rank 3, so it is outside the top 2.
    assert recall_at_k(["x", "y", "a"], ["a"], k=2) == 0.0
    assert recall_at_k(["x", "y", "a"], ["a"], k=3) == 1.0


def test_recall_of_a_query_with_no_expected_answer_is_zero_not_a_crash():
    assert recall_at_k(["a"], [], k=1) == 0.0


def test_duplicate_hits_do_not_inflate_recall():
    # Two chunks from the same source is one document found, not two.
    assert recall_at_k(["a", "a", "a"], ["a", "b"], k=3) == 0.5


def test_precision_measures_how_much_of_the_context_is_useful():
    assert precision_at_k(["a", "b", "c", "d"], ["a", "c"], k=4) == 0.5


def test_precision_of_an_empty_result_is_zero():
    assert precision_at_k([], ["a"], k=5) == 0.0


def test_hit_rate_is_binary():
    assert hit_rate_at_k(["a", "b"], ["b"], k=2) == 1.0
    assert hit_rate_at_k(["a", "b"], ["z"], k=2) == 0.0


def test_reciprocal_rank_rewards_ranking_the_answer_first():
    assert reciprocal_rank(["right", "wrong"], ["right"]) == 1.0
    assert reciprocal_rank(["wrong", "right"], ["right"]) == 0.5
    assert reciprocal_rank(["wrong", "wrong"], ["right"]) == 0.0


def test_mrr_separates_two_runs_that_have_identical_recall():
    # This is the case reranking is meant to fix: same documents found,
    # better order.
    before = ["x", "y", "a"]
    after = ["a", "x", "y"]

    assert recall_at_k(before, ["a"], k=3) == recall_at_k(after, ["a"], k=3)
    assert reciprocal_rank(after, ["a"]) > reciprocal_rank(before, ["a"])


def test_score_query_reports_every_metric():
    assert score_query(["a"], ["a"]) == {
        "recall": 1.0,
        "precision": 1.0,
        "hit_rate": 1.0,
        "mrr": 1.0,
    }


def test_mean_of_no_queries_is_zero_for_every_metric():
    assert mean_metrics([]) == {
        "recall": 0.0,
        "precision": 0.0,
        "hit_rate": 0.0,
        "mrr": 0.0,
    }


def test_mean_averages_across_queries():
    assert mean_metrics([{"recall": 1.0}, {"recall": 0.0}]) == {"recall": 0.5}


# --------------------------------------------------------------------------- #
# Golden sets
# --------------------------------------------------------------------------- #

def test_a_bare_list_is_a_valid_golden_set():
    golden = GoldenSet.from_data([{"query": "revenue?", "expected": ["a.pdf"]}])

    assert len(golden) == 1
    assert golden.queries[0].expected == ["a.pdf"]


def test_a_single_expected_file_does_not_have_to_be_a_list():
    golden = GoldenSet.from_data([{"query": "q", "expected": "a.pdf"}])
    assert golden.queries[0].expected == ["a.pdf"]


def test_settings_can_travel_with_the_queries():
    golden = GoldenSet.from_data({
        "match_field": "company_name",
        "match_mode": "exact",
        "queries": [{"query": "q", "expected": ["apple"]}],
    })

    assert golden.match_field == "company_name"
    assert golden.match_mode == "exact"


def test_a_query_with_no_expected_documents_is_rejected():
    # It would score 0.0 forever and quietly drag the average down.
    with pytest.raises(ValueError) as excinfo:
        GoldenSet.from_data([{"query": "q", "expected": []}])

    assert "expected" in str(excinfo.value)


def test_a_missing_expected_key_is_rejected():
    with pytest.raises(ValueError) as excinfo:
        GoldenSet.from_data([{"query": "q"}])

    assert "'expected'" in str(excinfo.value)


def test_an_empty_query_is_rejected():
    with pytest.raises(ValueError):
        GoldenSet.from_data([{"query": "   ", "expected": ["a.pdf"]}])


def test_unknown_match_mode_is_rejected_with_the_available_names():
    with pytest.raises(ValueError) as excinfo:
        GoldenSet([], match_mode="fuzzy")

    assert "basename" in str(excinfo.value)


def test_basename_matching_ignores_the_directory():
    golden = GoldenSet.from_data([{"query": "q", "expected": ["a.pdf"]}])
    doc = Document(page_content="", metadata={"source": "C:/data/filings/a.pdf"})

    assert golden.matches(golden.identify(doc), "a.pdf")


def test_basename_matching_is_case_insensitive():
    golden = GoldenSet.from_data([{"query": "q", "expected": ["A.PDF"]}])
    doc = Document(page_content="", metadata={"source": "/data/a.pdf"})

    assert golden.matches(golden.identify(doc), "A.PDF")


def test_exact_matching_compares_the_stored_value_verbatim():
    golden = GoldenSet([], match_mode="exact")
    doc = Document(page_content="", metadata={"source": "/data/a.pdf"})

    assert golden.matches(golden.identify(doc), "/data/a.pdf")
    assert not golden.matches(golden.identify(doc), "a.pdf")


def test_a_document_missing_the_match_field_never_matches():
    golden = GoldenSet.from_data([{"query": "q", "expected": ["a.pdf"]}])
    doc = Document(page_content="", metadata={})

    assert golden.identify(doc) == ""
    assert not golden.matches("", "a.pdf")


def test_golden_sets_load_from_json(tmp_path):
    path = tmp_path / "golden.json"
    path.write_text(json.dumps([{"query": "q", "expected": ["a.pdf"]}]), encoding="utf-8")

    assert len(GoldenSet.from_file(path)) == 1


def test_golden_sets_load_from_yaml(tmp_path):
    path = tmp_path / "golden.yaml"
    path.write_text(
        "- query: what is revenue\n  expected: [a.pdf]\n", encoding="utf-8"
    )

    golden = GoldenSet.from_file(path)
    assert golden.queries[0].query == "what is revenue"


def test_a_missing_golden_file_says_so():
    with pytest.raises(FileNotFoundError):
        GoldenSet.from_file("does_not_exist.yaml")


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #

def _docs(*sources):
    return [Document(page_content="", metadata={"source": s}) for s in sources]


def _fixed(*sources):
    """A retrieve() that always returns the same documents."""
    def retrieve(query, top_k=5, **kwargs):
        return _docs(*sources)[:top_k]
    return retrieve


GOLDEN = GoldenSet.from_data([
    {"query": "q1", "expected": ["a.pdf"]},
    {"query": "q2", "expected": ["b.pdf"]},
])


def test_a_perfect_run_scores_one():
    def retrieve(query, top_k=5, **kwargs):
        return _docs("a.pdf") if query == "q1" else _docs("b.pdf")

    result = evaluate(None, GOLDEN, top_k=5, retrieve=retrieve)

    assert result.metrics["recall"] == 1.0
    assert result.metrics["mrr"] == 1.0
    assert result.failures == []


def test_a_run_that_finds_nothing_scores_zero_and_lists_the_failures():
    result = evaluate(None, GOLDEN, top_k=5, retrieve=_fixed("z.pdf"))

    assert result.metrics["recall"] == 0.0
    assert len(result.failures) == 2
    assert result.per_query[0].missed == ["a.pdf"]


def test_half_the_queries_answered_averages_to_half():
    def retrieve(query, top_k=5, **kwargs):
        return _docs("a.pdf") if query == "q1" else _docs("z.pdf")

    result = evaluate(None, GOLDEN, top_k=5, retrieve=retrieve)
    assert result.metrics["recall"] == 0.5
    assert result.metrics["hit_rate"] == 0.5


def test_rank_position_is_reflected_in_mrr():
    result = evaluate(None, GOLDEN, top_k=5, retrieve=_fixed("z.pdf", "a.pdf", "b.pdf"))

    # Both queries find their document, so recall is perfect either way.
    assert result.metrics["recall"] == 1.0
    # a.pdf is at rank 2 and b.pdf at rank 3, so MRR averages 1/2 and 1/3.
    assert result.metrics["mrr"] == pytest.approx((0.5 + 1 / 3) / 2)


def test_a_failing_query_does_not_abort_the_whole_run():
    def retrieve(query, top_k=5, **kwargs):
        if query == "q1":
            raise ConnectionError("qdrant unavailable")
        return _docs("b.pdf")

    result = evaluate(None, GOLDEN, top_k=5, retrieve=retrieve)

    assert len(result.per_query) == 2
    assert result.metrics["recall"] == 0.5


def test_per_query_filters_reach_the_retriever():
    golden = GoldenSet.from_data([
        {"query": "q", "expected": ["a.pdf"], "filters": {"company_name": "apple"}},
    ])
    seen = {}

    def retrieve(query, top_k=5, **kwargs):
        seen.update(kwargs)
        return _docs("a.pdf")

    evaluate(None, golden, top_k=5, retrieve=retrieve)
    assert seen["filters"] == {"company_name": "apple"}


def test_retrieve_kwargs_are_passed_through():
    seen = {}

    def retrieve(query, top_k=5, **kwargs):
        seen.update(kwargs)
        return _docs("a.pdf")

    evaluate(None, GOLDEN, top_k=5, retrieve=retrieve, rerank=False)
    assert seen["rerank"] is False


def test_top_k_bounds_what_is_scored():
    # The correct document sits at rank 3 and must not count at top_k=2.
    result = evaluate(None, GOLDEN, top_k=2, retrieve=_fixed("x.pdf", "y.pdf", "a.pdf"))
    assert result.metrics["recall"] == 0.0


def test_the_table_names_the_run_and_its_cutoff():
    result = evaluate(None, GOLDEN, top_k=3, label="baseline", retrieve=_fixed("a.pdf"))
    table = result.to_table()

    assert "baseline" in table
    assert "top_k=3" in table
    assert "recall@3" in table


# --------------------------------------------------------------------------- #
# Sweeps
# --------------------------------------------------------------------------- #

def test_sweep_evaluates_every_variant():
    def retrieve(query, top_k=5, rerank=None, **kwargs):
        # Reranking moves the correct document from rank 2 to rank 1.
        return _docs("a.pdf", "z.pdf") if rerank else _docs("z.pdf", "a.pdf")

    result = sweep(
        None,
        GoldenSet.from_data([{"query": "q", "expected": ["a.pdf"]}]),
        {"baseline": {"rerank": False}, "reranked": {"rerank": True}},
        retrieve=retrieve,
    )

    assert [r.label for r in result.results] == ["baseline", "reranked"]
    assert result.results[0].metrics["mrr"] == 0.5
    assert result.results[1].metrics["mrr"] == 1.0


def test_sweep_variants_can_override_top_k():
    seen = []

    def retrieve(query, top_k=5, **kwargs):
        seen.append(top_k)
        return _docs("a.pdf")

    sweep(
        None,
        GoldenSet.from_data([{"query": "q", "expected": ["a.pdf"]}]),
        {"narrow": {"top_k": 3}, "wide": {"top_k": 10}},
        retrieve=retrieve,
    )

    assert seen == [3, 10]


def test_sweep_picks_the_best_variant_by_recall():
    def retrieve(query, top_k=5, rerank=None, **kwargs):
        return _docs("a.pdf") if rerank else _docs("z.pdf")

    result = sweep(
        None,
        GoldenSet.from_data([{"query": "q", "expected": ["a.pdf"]}]),
        {"baseline": {"rerank": False}, "reranked": {"rerank": True}},
        retrieve=retrieve,
    )

    assert result.best.label == "reranked"
    assert "Best recall: reranked" in result.to_table()


def test_an_empty_sweep_says_so_instead_of_crashing():
    from ragwire.eval import SweepResult

    assert SweepResult([]).best is None
    assert "No variants" in SweepResult([]).to_table()


def test_the_comparison_table_shows_deltas_against_the_first_variant():
    def retrieve(query, top_k=5, rerank=None, **kwargs):
        return _docs("a.pdf") if rerank else _docs("z.pdf")

    table = sweep(
        None,
        GoldenSet.from_data([{"query": "q", "expected": ["a.pdf"]}]),
        {"baseline": {"rerank": False}, "reranked": {"rerank": True}},
        retrieve=retrieve,
    ).to_table()

    # The second row carries a signed delta so the improvement is readable.
    assert "+1.00" in table
