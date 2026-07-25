"""
The objects a grounded answer is made of.

A RAG answer that is just a string is not much use in production: you cannot
show the user where it came from, you cannot audit it later, and you cannot
tell the difference between a confident answer and a guess. Answer carries the
text along with everything needed to check it.
"""

from typing import Any, Dict, List, Optional

# The exact string the model is told to return when the sources do not answer
# the question. Kept as a sentinel rather than a phrase so that detecting a
# refusal never depends on matching prose.
REFUSAL_SENTINEL = "INSUFFICIENT_CONTEXT"


class Citation:
    """
    One source backing an answer.

    Attributes:
        index: The number the answer refers to it by, as in ``[2]``
        source: The file the chunk came from
        text: The chunk text that was shown to the model
        metadata: The chunk's full metadata
        score: The rerank score, when reranking produced one
    """

    def __init__(
        self,
        index: int,
        source: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        score: Optional[float] = None,
    ):
        self.index = index
        self.source = source
        self.text = text
        self.metadata = metadata or {}
        self.score = score

    @property
    def snippet(self) -> str:
        """The first 200 characters, for showing next to an answer."""
        text = " ".join(self.text.split())
        return text[:200] + ("..." if len(text) > 200 else "")

    def to_dict(self) -> Dict[str, Any]:
        """Serialise for logging or an API response."""
        return {
            "index": self.index,
            "source": self.source,
            "snippet": self.snippet,
            "score": self.score,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        return f"Citation([{self.index}] {self.source})"


class Answer:
    """
    A generated answer and the evidence behind it.

    Attributes:
        text: The answer, or an explanation of why none could be given
        citations: The sources the answer actually cites, in the order first
            referenced. Sources that were retrieved but never cited are not
            listed here; see ``documents`` for the full candidate set.
        documents: Every document that was retrieved for the question
        filters_used: Metadata filters applied during retrieval, if any
        refused: True when the model reported that the sources do not answer
            the question, or when nothing was retrieved at all
        confidence: The fraction of the answer's sentences that carry a
            citation. See the note below.

    !!! warning "What confidence measures"
        ``confidence`` is citation coverage, not a probability of being
        correct. A fully cited answer drawn from a chunk that happens to be
        wrong will still score 1.0. It tells you how much of the answer is
        traceable to a source, which is a useful signal and not the same thing
        as accuracy.
    """

    def __init__(
        self,
        text: str,
        citations: Optional[List[Citation]] = None,
        documents: Optional[List[Any]] = None,
        filters_used: Optional[Dict[str, Any]] = None,
        refused: bool = False,
        confidence: float = 0.0,
        query: str = "",
    ):
        self.text = text
        self.citations = citations or []
        self.documents = documents or []
        self.filters_used = filters_used
        self.refused = refused
        self.confidence = confidence
        self.query = query

    @property
    def sources(self) -> List[str]:
        """The distinct files cited, in the order first referenced."""
        seen = []
        for citation in self.citations:
            if citation.source not in seen:
                seen.append(citation.source)
        return seen

    def to_dict(self) -> Dict[str, Any]:
        """Serialise for logging or an API response."""
        return {
            "query": self.query,
            "text": self.text,
            "refused": self.refused,
            "confidence": self.confidence,
            "filters_used": self.filters_used,
            "citations": [c.to_dict() for c in self.citations],
        }

    def formatted(self) -> str:
        """
        The answer with a numbered source list appended.

        Returns:
            Answer text followed by the citations, ready to print

        Example:
            >>> print(answer.formatted())  # doctest: +SKIP
            Revenue grew 12% year over year [1].
            <BLANKLINE>
            Sources:
              [1] apple_10k_2025.pdf
        """
        if not self.citations:
            return self.text

        lines = [self.text, "", "Sources:"]
        for citation in self.citations:
            lines.append(f"  [{citation.index}] {citation.source}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        state = "refused" if self.refused else f"{len(self.citations)} citations"
        return f"<Answer {state}, confidence={self.confidence:.2f}>"

    def __bool__(self) -> bool:
        """An answer is falsy when the sources did not support one."""
        return not self.refused
