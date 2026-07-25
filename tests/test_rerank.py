"""
Tests for second-stage reranking.

No model downloads, no network. The reranker providers are exercised through
a stub that scores deterministically, so these tests cover the wiring rather
than the quality of any particular model.
"""

import pytest
from langchain_core.documents import Document

from ragwire import RAGWire
from ragwire.retriever.rerank import (
    BaseReranker,
    get_reranker,
    resolve_fetch_k,
)


class _ScoreByKeyword(BaseReranker):
    """Scores a document by how many times a keyword appears in it."""

    name = "stub"

    def __init__(self, keyword="match"):
        self.keyword = keyword
        self.calls = []

    def _score(self, query, texts):
        self.calls.append((query, list(texts)))
        return [float(t.count(self.keyword)) for t in texts]


class _WrongLengthReranker(BaseReranker):
    name = "broken"

    def _score(self, query, texts):
        return [1.0]


def _docs(*contents):
    return [Document(page_content=c, metadata={"source": f"{i}.pdf"})
            for i, c in enumerate(contents)]


# --------------------------------------------------------------------------- #
# get_reranker: the off switch
# --------------------------------------------------------------------------- #

def test_no_config_means_no_reranker():
    assert get_reranker(None) is None
    assert get_reranker({}) is None


def test_enabled_false_keeps_settings_but_turns_reranking_off():
    config = {"enabled": False, "provider": "cross_encoder", "fetch_k": 40}
    assert get_reranker(config) is None


def test_unknown_provider_is_rejected_with_the_available_names():
    with pytest.raises(ValueError) as excinfo:
        get_reranker({"provider": "magic"})

    message = str(excinfo.value)
    assert "magic" in message
    assert "cross_encoder" in message and "cohere" in message


# --------------------------------------------------------------------------- #
# resolve_fetch_k: the candidate pool
# --------------------------------------------------------------------------- #

def test_no_reranking_fetches_exactly_what_the_caller_asked_for():
    assert resolve_fetch_k(None, top_k=5) == 5


def test_default_pool_is_four_times_top_k_with_a_floor():
    assert resolve_fetch_k({"provider": "cohere"}, top_k=10) == 40
    # 4 * 2 is too small a pool to rerank meaningfully, so the floor applies.
    assert resolve_fetch_k({"provider": "cohere"}, top_k=2) == 20


def test_explicit_fetch_k_wins():
    assert resolve_fetch_k({"fetch_k": 50}, top_k=5) == 50


def test_fetch_k_below_top_k_would_shrink_the_result_set():
    # Asking for 10 documents while reranking only 3 candidates should not
    # quietly return 3.
    assert resolve_fetch_k({"fetch_k": 3}, top_k=10) == 10


# --------------------------------------------------------------------------- #
# BaseReranker: ordering, truncation, scores
# --------------------------------------------------------------------------- #

def test_documents_are_reordered_by_score():
    reranker = _ScoreByKeyword()
    docs = _docs("nothing here", "match match match", "match once")

    ranked = reranker.rerank("q", docs)

    assert [d.page_content for d in ranked] == [
        "match match match",
        "match once",
        "nothing here",
    ]


def test_top_n_truncates_after_reordering():
    reranker = _ScoreByKeyword()
    docs = _docs("nothing", "match match", "match")

    ranked = reranker.rerank("q", docs, top_n=2)

    # The best two survive, not the first two.
    assert [d.page_content for d in ranked] == ["match match", "match"]


def test_scores_are_attached_to_the_returned_documents():
    reranker = _ScoreByKeyword()
    ranked = reranker.rerank("q", _docs("match match", "nothing"))

    assert ranked[0].metadata["rerank_score"] == 2.0
    assert ranked[1].metadata["rerank_score"] == 0.0


def test_empty_candidate_list_short_circuits():
    reranker = _ScoreByKeyword()
    assert reranker.rerank("q", []) == []
    assert reranker.calls == []


def test_single_candidate_skips_the_model_call():
    reranker = _ScoreByKeyword()
    docs = _docs("only one")

    assert reranker.rerank("q", docs) == docs
    # Scoring one document cannot change its position, so the call is wasted.
    assert reranker.calls == []


def test_score_count_mismatch_is_an_error_not_a_silent_truncation():
    with pytest.raises(ValueError) as excinfo:
        _WrongLengthReranker().rerank("q", _docs("a", "b", "c"))

    assert "1 scores" in str(excinfo.value)
    assert "3 documents" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# Pipeline wiring
# --------------------------------------------------------------------------- #

class _StubRetriever:
    def __init__(self, search_kwargs, search_type="similarity"):
        self.search_kwargs = search_kwargs
        self.search_type = search_type


class _StubVectorStore:
    """Returns k numbered documents and records the kwargs it was given."""

    def __init__(self):
        self.seen_kwargs = None

    def as_retriever(self, search_type, search_kwargs):
        self.seen_kwargs = dict(search_kwargs)
        store = self

        class _R:
            def invoke(self, query):
                k = store.seen_kwargs["k"]
                # Later documents contain more matches, so a keyword reranker
                # reverses the vector store's ordering.
                return [
                    Document(page_content=" ".join(["match"] * i), metadata={"i": i})
                    for i in range(k)
                ]

        return _R()


def _pipeline(reranker=None, rerank_config=None, search_kwargs=None, search_type="similarity"):
    rag = object.__new__(RAGWire)
    rag.config = {"retriever": {"top_k": 3}}
    rag._auto_filter = False
    rag.vectorstore = _StubVectorStore()
    rag.retriever = _StubRetriever(search_kwargs or {"k": 3}, search_type)
    rag._rerank_config = rerank_config or {}
    rag.reranker = reranker
    return rag


def test_without_a_reranker_the_pool_is_not_widened():
    rag = _pipeline()

    results = rag.retrieve("q", top_k=3)

    assert rag.vectorstore.seen_kwargs["k"] == 3
    assert len(results) == 3


def test_reranking_fetches_wide_then_cuts_to_top_k():
    reranker = _ScoreByKeyword()
    rag = _pipeline(reranker, {"fetch_k": 12})

    results = rag.retrieve("q", top_k=3)

    assert rag.vectorstore.seen_kwargs["k"] == 12
    assert len(results) == 3
    # The stub store returns increasing match counts, so reranking must invert it.
    assert [d.metadata["i"] for d in results] == [11, 10, 9]


def test_rerank_false_skips_reranking_for_one_call():
    reranker = _ScoreByKeyword()
    rag = _pipeline(reranker, {"fetch_k": 12})

    results = rag.retrieve("q", top_k=3, rerank=False)

    assert rag.vectorstore.seen_kwargs["k"] == 3
    assert reranker.calls == []
    assert [d.metadata["i"] for d in results] == [0, 1, 2]


def test_rerank_true_without_a_configured_reranker_fails_loudly():
    rag = _pipeline(reranker=None)

    with pytest.raises(ValueError) as excinfo:
        rag.retrieve("q", rerank=True)

    assert "retriever.rerank" in str(excinfo.value)


def test_mmr_pool_grows_to_cover_the_candidate_count():
    reranker = _ScoreByKeyword()
    rag = _pipeline(
        reranker,
        {"fetch_k": 30},
        search_kwargs={"k": 3, "fetch_k": 20, "lambda_mult": 0.5},
        search_type="mmr",
    )

    rag.retrieve("q", top_k=3)

    # MMR selects k out of its own fetch_k pool, so a pool of 20 could not
    # supply the 30 candidates the reranker was asked to score.
    assert rag.vectorstore.seen_kwargs["k"] == 30
    assert rag.vectorstore.seen_kwargs["fetch_k"] == 30


def test_the_query_reaches_the_reranker_unchanged():
    reranker = _ScoreByKeyword()
    rag = _pipeline(reranker, {"fetch_k": 5})

    rag.retrieve("what was the revenue", top_k=2)

    assert reranker.calls[0][0] == "what was the revenue"
