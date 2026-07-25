"""
Grounded answer generation.

Retrieval hands back chunks. This module turns them into an answer that cites
its sources and refuses when the sources do not support one.

Example:
    >>> from ragwire import RAGWire
    >>> rag = RAGWire("config.yaml")                        # doctest: +SKIP
    >>> answer = rag.query("What was net income in 2025?")  # doctest: +SKIP
    >>> print(answer.formatted())                           # doctest: +SKIP
"""

from .answer import REFUSAL_SENTINEL, Answer, Citation
from .generator import (
    DEFAULT_SYSTEM_PROMPT,
    AnswerGenerator,
    build_context,
    citation_coverage,
    parse_citations,
)

__all__ = [
    "Answer",
    "Citation",
    "AnswerGenerator",
    "DEFAULT_SYSTEM_PROMPT",
    "REFUSAL_SENTINEL",
    "build_context",
    "citation_coverage",
    "parse_citations",
]
