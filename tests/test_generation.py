"""
Tests for grounded answer generation.

The LLM is a stub that returns whatever it was told to, so these cover
prompting, citation parsing, refusal handling and the context budget rather
than the quality of any particular model.
"""

import pytest
from langchain_core.documents import Document

from ragwire import RAGWire
from ragwire.generation import (
    REFUSAL_SENTINEL,
    Answer,
    AnswerGenerator,
    Citation,
    build_context,
    citation_coverage,
    parse_citations,
)
from ragwire.generation.generator import NO_CONTEXT_MESSAGE, REFUSAL_MESSAGE


class _StubResponse:
    def __init__(self, content):
        self.content = content


class _StubLLM:
    """Returns a canned answer and records the messages it was sent."""

    def __init__(self, reply="An answer [1]."):
        self.reply = reply
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return _StubResponse(self.reply)

    async def ainvoke(self, messages):
        self.messages = messages
        return _StubResponse(self.reply)


def _docs(*sources, text="chunk text"):
    return [
        Document(page_content=f"{text} {s}", metadata={"source": s})
        for s in sources
    ]


# --------------------------------------------------------------------------- #
# Context building
# --------------------------------------------------------------------------- #

def test_sources_are_numbered_from_one():
    context, used = build_context(_docs("a.pdf", "b.pdf"))

    assert "[1] (source: a.pdf)" in context
    assert "[2] (source: b.pdf)" in context
    assert len(used) == 2


def test_a_document_with_no_source_is_still_citable():
    context, used = build_context([Document(page_content="text", metadata={})])

    assert "(source: unknown)" in context
    assert len(used) == 1


def test_the_character_budget_drops_chunks_that_do_not_fit():
    docs = [Document(page_content="x" * 500, metadata={"source": f"{i}.pdf"})
            for i in range(10)]

    context, used = build_context(docs, max_context_chars=1200)

    # Two chunks fit inside the budget and the remaining eight are dropped.
    assert len(used) == 2
    assert len(context) <= 1200


def test_a_chunk_crossing_the_budget_is_truncated_rather_than_dropped():
    docs = [Document(page_content="y" * 5000, metadata={"source": "big.pdf"})]

    context, used = build_context(docs, max_context_chars=1000)

    assert len(used) == 1
    assert "[truncated]" in context


def test_only_the_documents_that_fit_may_be_cited():
    docs = [Document(page_content="z" * 400, metadata={"source": f"{i}.pdf"})
            for i in range(5)]

    _, used = build_context(docs, max_context_chars=900)

    # Citing a document the model never saw would point the reader at the
    # wrong file, so the returned list is the one callers must number against.
    assert len(used) == 2


def test_a_sliver_of_a_chunk_is_dropped_rather_than_given_a_source_number():
    docs = [
        Document(page_content="a" * 400, metadata={"source": "first.pdf"}),
        Document(page_content="b" * 400, metadata={"source": "second.pdf"}),
    ]

    # Leaves roughly 60 characters for the second chunk, which is too little
    # to be worth a source number.
    _, used = build_context(docs, max_context_chars=500)

    assert [d.metadata["source"] for d in used] == ["first.pdf"]


def test_the_budget_is_never_exceeded_by_the_truncation_marker():
    docs = [Document(page_content="q" * 10000, metadata={"source": "big.pdf"})]

    context, _ = build_context(docs, max_context_chars=1000)

    assert len(context) <= 1000


# --------------------------------------------------------------------------- #
# Citation parsing
# --------------------------------------------------------------------------- #

def test_citations_resolve_to_the_numbered_document():
    docs = _docs("a.pdf", "b.pdf")

    text, citations = parse_citations("Revenue grew [2].", docs)

    assert len(citations) == 1
    assert citations[0].index == 2
    assert citations[0].source == "b.pdf"


def test_citations_are_returned_in_order_of_first_appearance():
    docs = _docs("a.pdf", "b.pdf", "c.pdf")

    _, citations = parse_citations("Claim [3]. Another [1]. Again [3].", docs)

    assert [c.index for c in citations] == [3, 1]


def test_repeated_citations_are_not_duplicated():
    docs = _docs("a.pdf")

    _, citations = parse_citations("One [1]. Two [1]. Three [1].", docs)

    assert len(citations) == 1


def test_a_citation_pointing_past_the_sources_is_stripped():
    docs = _docs("a.pdf")

    text, citations = parse_citations("Revenue grew [7].", docs)

    # Showing a reader [7] with no seventh source is worse than showing none.
    assert "[7]" not in text
    assert citations == []


def test_uncited_answers_produce_no_citations():
    _, citations = parse_citations("Revenue grew.", _docs("a.pdf"))
    assert citations == []


def test_the_rerank_score_travels_onto_the_citation():
    doc = Document(page_content="t", metadata={"source": "a.pdf", "rerank_score": 4.5})

    _, citations = parse_citations("Claim [1].", [doc])

    assert citations[0].score == 4.5


# --------------------------------------------------------------------------- #
# Citation coverage
# --------------------------------------------------------------------------- #

def test_a_fully_cited_answer_scores_one():
    assert citation_coverage("Revenue grew [1]. Margins fell [2].") == 1.0


def test_a_half_cited_answer_scores_half():
    assert citation_coverage("Revenue grew [1]. Margins fell.") == 0.5


def test_an_uncited_answer_scores_zero():
    assert citation_coverage("Revenue grew.") == 0.0


def test_an_empty_answer_scores_zero_rather_than_dividing_by_nothing():
    assert citation_coverage("   ") == 0.0


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #

def test_an_answer_carries_its_citations_and_coverage():
    generator = AnswerGenerator(_StubLLM("Net income was $93.7bn [1]."))

    answer = generator.generate("net income?", _docs("apple.pdf"))

    assert answer.refused is False
    assert answer.confidence == 1.0
    assert answer.sources == ["apple.pdf"]


def test_retrieving_nothing_refuses_without_calling_the_model():
    llm = _StubLLM()
    generator = AnswerGenerator(llm)

    answer = generator.generate("anything?", [])

    assert answer.refused is True
    assert answer.text == NO_CONTEXT_MESSAGE
    # Sending an empty context to the model wastes a call to learn nothing.
    assert llm.messages is None


def test_the_refusal_sentinel_becomes_a_readable_message():
    generator = AnswerGenerator(_StubLLM(REFUSAL_SENTINEL))

    answer = generator.generate("what is the capital of France?", _docs("a.pdf"))

    assert answer.refused is True
    assert answer.text == REFUSAL_MESSAGE
    assert REFUSAL_SENTINEL not in answer.text
    assert answer.confidence == 0.0


def test_a_sentinel_wrapped_in_a_sentence_still_counts_as_a_refusal():
    generator = AnswerGenerator(_StubLLM(f"{REFUSAL_SENTINEL} - not in the sources."))

    assert generator.generate("q", _docs("a.pdf")).refused is True


def test_an_empty_model_response_is_treated_as_a_refusal():
    generator = AnswerGenerator(_StubLLM("   "))

    assert generator.generate("q", _docs("a.pdf")).refused is True


def test_a_long_answer_quoting_the_sentinel_is_not_a_refusal():
    reply = (
        "The filing uses the phrase " + REFUSAL_SENTINEL + " in its risk "
        "disclosures, which the auditors flagged as unusual boilerplate "
        "language throughout the document [1]."
    )
    generator = AnswerGenerator(_StubLLM(reply))

    assert generator.generate("q", _docs("a.pdf")).refused is False


def test_the_prompt_forbids_answering_from_general_knowledge():
    llm = _StubLLM()
    AnswerGenerator(llm).generate("q", _docs("a.pdf"))

    system = llm.messages[0][1]
    assert "general knowledge" in system
    assert REFUSAL_SENTINEL in system
    assert "{sentinel}" not in system


def test_the_question_and_sources_reach_the_model():
    llm = _StubLLM()
    AnswerGenerator(llm).generate("what is revenue?", _docs("a.pdf"))

    human = llm.messages[1][1]
    assert "what is revenue?" in human
    assert "[1] (source: a.pdf)" in human


def test_a_custom_system_prompt_replaces_the_default():
    llm = _StubLLM()
    AnswerGenerator(llm, system_prompt="Answer in French. Refuse with {sentinel}.")\
        .generate("q", _docs("a.pdf"))

    system = llm.messages[0][1]
    assert "Answer in French" in system
    assert REFUSAL_SENTINEL in system


def test_a_custom_prompt_containing_braces_does_not_break_formatting():
    llm = _StubLLM()
    # A prompt with a JSON example would raise if str.format were used.
    AnswerGenerator(llm, system_prompt='Reply as {"a": 1}.').generate("q", _docs("a.pdf"))

    assert '{"a": 1}' in llm.messages[0][1]


def test_filters_used_are_recorded_on_the_answer():
    generator = AnswerGenerator(_StubLLM())

    answer = generator.generate("q", _docs("a.pdf"), filters_used={"company_name": "apple"})

    assert answer.filters_used == {"company_name": "apple"}


@pytest.mark.asyncio
async def test_agenerate_matches_generate():
    generator = AnswerGenerator(_StubLLM("Answer [1]."))

    answer = await generator.agenerate("q", _docs("a.pdf"))

    assert answer.refused is False
    assert answer.citations[0].source == "a.pdf"


# --------------------------------------------------------------------------- #
# Answer object
# --------------------------------------------------------------------------- #

def test_a_refused_answer_is_falsy():
    assert not Answer("no", refused=True)
    assert Answer("yes", refused=False)


def test_formatted_appends_a_numbered_source_list():
    answer = Answer(
        "Revenue grew [1].",
        citations=[Citation(1, "apple.pdf", "text")],
    )

    output = answer.formatted()
    assert "Sources:" in output
    assert "[1] apple.pdf" in output


def test_formatted_is_just_the_text_when_nothing_was_cited():
    assert Answer("Revenue grew.").formatted() == "Revenue grew."


def test_sources_are_deduplicated_in_first_reference_order():
    answer = Answer(
        "text",
        citations=[
            Citation(1, "a.pdf", "t"),
            Citation(2, "b.pdf", "t"),
            Citation(3, "a.pdf", "t"),
        ],
    )

    assert answer.sources == ["a.pdf", "b.pdf"]


def test_a_citation_snippet_is_bounded_and_collapsed():
    citation = Citation(1, "a.pdf", "word  \n  spaced " + "x" * 400)

    assert len(citation.snippet) <= 203
    assert "  " not in citation.snippet


def test_an_answer_serialises_for_logging():
    answer = Answer("text [1].", citations=[Citation(1, "a.pdf", "t")], query="q")
    data = answer.to_dict()

    assert data["query"] == "q"
    assert data["citations"][0]["source"] == "a.pdf"


# --------------------------------------------------------------------------- #
# Pipeline wiring
# --------------------------------------------------------------------------- #

def _pipeline(llm, auto_filter=False, extracted=None, retrieved=None):
    rag = object.__new__(RAGWire)
    rag.config = {"retriever": {"top_k": 3}}
    rag._auto_filter = auto_filter
    rag.generator = AnswerGenerator(llm)
    rag._extract_calls = 0

    def _extract(query):
        rag._extract_calls += 1
        return extracted

    def _retrieve(query, top_k=None, filters=None, rerank=None):
        rag._retrieve_kwargs = {"top_k": top_k, "filters": filters, "rerank": rerank}
        return retrieved if retrieved is not None else _docs("a.pdf")

    rag._extract_filters_from_query = _extract
    rag.retrieve = _retrieve
    return rag


def test_query_returns_an_answer_grounded_in_retrieval():
    rag = _pipeline(_StubLLM("Net income was $93.7bn [1]."))

    answer = rag.query("net income?")

    assert answer.text == "Net income was $93.7bn [1]."
    assert answer.citations[0].source == "a.pdf"


def test_explicit_filters_are_passed_through_and_reported():
    rag = _pipeline(_StubLLM(), auto_filter=True)

    answer = rag.query("q", filters={"company_name": "apple"})

    assert rag._retrieve_kwargs["filters"] == {"company_name": "apple"}
    assert answer.filters_used == {"company_name": "apple"}
    # Explicit filters mean there is nothing to extract.
    assert rag._extract_calls == 0


def test_auto_filter_extracts_once_and_records_the_result():
    rag = _pipeline(_StubLLM(), auto_filter=True, extracted={"fiscal_year": 2025})

    answer = rag.query("2025 revenue?")

    assert rag._extract_calls == 1
    assert answer.filters_used == {"fiscal_year": 2025}


def test_failed_extraction_does_not_trigger_a_second_llm_call():
    rag = _pipeline(_StubLLM(), auto_filter=True, extracted=None)

    answer = rag.query("q")

    # An empty dict rather than None is what stops retrieve() re-extracting.
    assert rag._retrieve_kwargs["filters"] == {}
    assert rag._extract_calls == 1
    assert answer.filters_used is None


def test_query_forwards_the_rerank_override():
    rag = _pipeline(_StubLLM())

    rag.query("q", rerank=False, top_k=7)

    assert rag._retrieve_kwargs["rerank"] is False
    assert rag._retrieve_kwargs["top_k"] == 7


def test_query_refuses_when_retrieval_comes_back_empty():
    rag = _pipeline(_StubLLM(), retrieved=[])

    answer = rag.query("q")

    assert answer.refused is True
    assert not answer


@pytest.mark.asyncio
async def test_aquery_matches_query():
    rag = _pipeline(_StubLLM("Answer [1]."))

    answer = await rag.aquery("q")

    assert answer.refused is False
    assert answer.citations[0].source == "a.pdf"
