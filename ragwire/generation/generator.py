"""
Turning retrieved chunks into a grounded answer.

The hard part of RAG generation is not producing fluent text, it is making the
model stay inside the sources it was given and admit when they do not answer
the question. Both are handled here: sources are numbered so every claim can
point at one, and the model is given an explicit way to say no.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from .answer import REFUSAL_SENTINEL, Answer, Citation

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = """You answer questions using only the numbered sources provided.

Rules:
1. Use only the sources. Never answer from general knowledge, even when you are confident.
2. Cite every factual claim with its source number in square brackets, like [2]. Cite more than one where more than one applies, like [1][3].
3. If the sources do not contain the answer, reply with exactly {sentinel} and nothing else. Do not guess, and do not offer a partial answer built on what is missing.
4. Quote figures, dates and names exactly as they appear in the sources.
5. Answer directly. Do not mention these rules, the sources as a concept, or the fact that you were given context."""

NO_CONTEXT_MESSAGE = (
    "No documents were retrieved for this question, so there is nothing to "
    "answer from."
)

REFUSAL_MESSAGE = (
    "The retrieved documents do not contain enough information to answer this "
    "question."
)

_CITATION_PATTERN = re.compile(r"\[(\d+)\]")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

BLOCK_SEPARATOR = "\n\n"
TRUNCATION_MARKER = "\n[truncated]"

# Below this many characters a chunk carries too little to be worth a source
# number, so the context block ends instead.
MIN_CHUNK_CHARS = 200


def build_context(
    documents: List[Any], max_context_chars: int = 12000
) -> Tuple[str, List[Any]]:
    """
    Render retrieved documents as a numbered source block.

    Chunks are added in rank order until the character budget runs out. This
    matters because a default RAGWire chunk is 10,000 characters and five of
    them will overflow most context windows, producing either an API error or
    a silent truncation that drops the best chunk last.

    Args:
        documents: Retrieved documents, best first
        max_context_chars: Total budget across all chunks

    Returns:
        The rendered context, and the documents that actually fit. Only the
        second list may be cited, so callers must use it rather than the input.

    Example:
        >>> context, used = build_context(docs, max_context_chars=4000)  # doctest: +SKIP
    """
    blocks = []
    used: List[Any] = []
    spent = 0

    for document in documents:
        source = document.metadata.get("source", "unknown")
        header = f"[{len(used) + 1}] (source: {source})\n"
        separator = len(BLOCK_SEPARATOR) if used else 0
        remaining = max_context_chars - spent - len(header) - separator

        text = document.page_content

        if len(text) > remaining:
            # A truncated chunk is still worth including, since the top of a
            # chunk usually carries the heading that makes it citable. A sliver
            # of one is not: it wastes a source number the model could have
            # spent on a chunk it can actually use.
            keep = remaining - len(TRUNCATION_MARKER)
            if keep < MIN_CHUNK_CHARS:
                break
            text = text[:keep].rstrip() + TRUNCATION_MARKER

        blocks.append(header + text)
        used.append(document)
        spent += separator + len(header) + len(text)

    if len(used) < len(documents):
        logger.info(
            f"Context budget of {max_context_chars} chars fit {len(used)} of "
            f"{len(documents)} retrieved chunks"
        )

    return BLOCK_SEPARATOR.join(blocks), used


def parse_citations(text: str, documents: List[Any]) -> Tuple[str, List[Citation]]:
    """
    Find the source numbers an answer refers to and resolve them.

    Args:
        text: The generated answer
        documents: The documents that were shown to the model, in the order
            they were numbered

    Returns:
        The answer text, and the citations it references in order of first
        appearance. Markers pointing at a source that does not exist are
        dropped from the text, since showing a reader ``[7]`` with no seventh
        source is worse than showing nothing.
    """
    citations: List[Citation] = []
    seen: Dict[int, Citation] = {}
    invalid: List[str] = []

    for match in _CITATION_PATTERN.finditer(text):
        number = int(match.group(1))

        if not 1 <= number <= len(documents):
            invalid.append(match.group(0))
            continue

        if number in seen:
            continue

        document = documents[number - 1]
        citation = Citation(
            index=number,
            source=document.metadata.get("source", "unknown"),
            text=document.page_content,
            metadata=dict(document.metadata),
            score=document.metadata.get("rerank_score"),
        )
        seen[number] = citation
        citations.append(citation)

    if invalid:
        logger.warning(
            f"Answer referenced {len(invalid)} source numbers that do not "
            f"exist: {', '.join(sorted(set(invalid)))}"
        )
        for marker in set(invalid):
            text = text.replace(marker, "")
        text = re.sub(r"[ ]{2,}", " ", text).strip()

    return text, citations


def citation_coverage(text: str) -> float:
    """
    The fraction of an answer's sentences that carry a citation.

    This is a groundedness signal, not an accuracy one. It answers "how much of
    this can the reader trace back to a source", which is the question you can
    actually check automatically.

    Args:
        text: The generated answer

    Returns:
        A value from 0.0 to 1.0. Returns 0.0 for an empty answer.

    Example:
        >>> citation_coverage("Revenue grew [1]. Margins fell [2].")
        1.0
        >>> citation_coverage("Revenue grew [1]. Margins fell.")
        0.5
    """
    sentences = [s for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()]
    if not sentences:
        return 0.0

    cited = sum(1 for s in sentences if _CITATION_PATTERN.search(s))
    return cited / len(sentences)


def _is_refusal(text: str) -> bool:
    """
    Decide whether the model declined to answer.

    Models often wrap the sentinel in a sentence rather than returning it
    alone, so a bare containment check is more reliable than equality. The
    length guard stops a real answer that happens to quote the sentinel from
    being read as a refusal.
    """
    stripped = text.strip()
    if not stripped:
        return True
    return REFUSAL_SENTINEL in stripped and len(stripped) < len(REFUSAL_SENTINEL) + 120


class AnswerGenerator:
    """
    Generates grounded answers from retrieved documents.

    Holds no retrieval logic of its own. It is given documents and produces an
    Answer, which keeps it usable with any retrieval strategy and trivial to
    test without a vector store.

    Attributes:
        llm: A LangChain chat model
        max_context_chars: Total character budget for the source block
        system_prompt: The instruction block sent before the sources
    """

    def __init__(
        self,
        llm: Any,
        max_context_chars: int = 12000,
        system_prompt: Optional[str] = None,
    ):
        self.llm = llm
        self.max_context_chars = max_context_chars

        # A plain replace rather than str.format, so a custom prompt containing
        # braces (a JSON example, say) does not raise.
        prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.system_prompt = prompt.replace("{sentinel}", REFUSAL_SENTINEL)

    def build_messages(self, query: str, context: str) -> List[Tuple[str, str]]:
        """
        Assemble the chat messages for a question.

        Args:
            query: The user's question
            context: The rendered source block from :func:`build_context`

        Returns:
            LangChain-style (role, content) message tuples
        """
        return [
            ("system", self.system_prompt),
            ("human", f"Sources:\n\n{context}\n\nQuestion: {query}"),
        ]

    def generate(
        self,
        query: str,
        documents: List[Any],
        filters_used: Optional[Dict[str, Any]] = None,
    ) -> Answer:
        """
        Answer a question from retrieved documents.

        Args:
            query: The question
            documents: Retrieved documents, best first
            filters_used: Filters that produced them, recorded on the Answer

        Returns:
            An Answer. Never raises for an unanswerable question: it returns a
            refusal instead, because "I do not know" is a valid result and
            callers should not have to catch an exception to handle it.
        """
        if not documents:
            return self._empty(query, filters_used)

        context, used = build_context(documents, self.max_context_chars)
        response = self.llm.invoke(self.build_messages(query, context))
        return self._finalize(query, response, used, documents, filters_used)

    async def agenerate(
        self,
        query: str,
        documents: List[Any],
        filters_used: Optional[Dict[str, Any]] = None,
    ) -> Answer:
        """
        Async version of :meth:`generate`.

        Args:
            query: The question
            documents: Retrieved documents, best first
            filters_used: Filters that produced them, recorded on the Answer

        Returns:
            An Answer
        """
        if not documents:
            return self._empty(query, filters_used)

        context, used = build_context(documents, self.max_context_chars)
        response = await self.llm.ainvoke(self.build_messages(query, context))
        return self._finalize(query, response, used, documents, filters_used)

    @staticmethod
    def _empty(query: str, filters_used: Optional[Dict[str, Any]]) -> Answer:
        """Nothing was retrieved, so there is nothing to send to the model."""
        return Answer(
            text=NO_CONTEXT_MESSAGE,
            documents=[],
            filters_used=filters_used,
            refused=True,
            confidence=0.0,
            query=query,
        )

    @staticmethod
    def _finalize(
        query: str,
        response: Any,
        used: List[Any],
        documents: List[Any],
        filters_used: Optional[Dict[str, Any]],
    ) -> Answer:
        """Turn a raw model response into an Answer."""
        text = getattr(response, "content", response)
        if not isinstance(text, str):
            text = str(text)
        text = text.strip()

        if _is_refusal(text):
            logger.info(f"Model refused to answer from the retrieved context: {query[:60]}")
            return Answer(
                text=REFUSAL_MESSAGE,
                documents=documents,
                filters_used=filters_used,
                refused=True,
                confidence=0.0,
                query=query,
            )

        text, citations = parse_citations(text, used)

        return Answer(
            text=text,
            citations=citations,
            documents=documents,
            filters_used=filters_used,
            refused=False,
            confidence=citation_coverage(text),
            query=query,
        )
