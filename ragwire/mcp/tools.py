"""
The tool implementations behind the MCP server.

These are plain functions over a RAGWire instance, deliberately separate from
the MCP framework that exposes them. That keeps them testable without the
optional ``mcp`` dependency, and it means the same functions can back a REST
endpoint or a LangChain tool without being rewritten.

Every function returns a string, because the caller on the other end is a
language model rather than code. Text with clear structure is easier for a
model to use correctly than JSON it has to parse and re-summarise.
"""

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MAX_SNIPPET_CHARS = 1500

# Metadata keys that describe how a chunk was stored rather than what it says.
# Showing them to an agent adds noise and invites it to filter on fields that
# were never meant to be query surface.
INTERNAL_METADATA_KEYS = {
    "_id",
    "_collection_name",
    "content_hash",
    "file_hash",
    "chunk_index",
    "total_chunks",
    "metadata_status",
}


def _parse_filters(filters: Optional[Any]) -> Optional[Dict[str, Any]]:
    """
    Accept filters as either a mapping or a JSON string.

    Models frequently send a JSON string where a schema asked for an object.
    Rejecting that would be technically correct and practically useless, so it
    is parsed instead.
    """
    if filters is None or filters == "":
        return None
    if isinstance(filters, dict):
        return filters or None
    if isinstance(filters, str):
        try:
            parsed = json.loads(filters)
        except json.JSONDecodeError:
            raise ValueError(
                f"filters must be a JSON object, got unparseable string: {filters[:80]}"
            )
        if not isinstance(parsed, dict):
            raise ValueError(f"filters must be a JSON object, got {type(parsed).__name__}")
        return parsed or None

    raise ValueError(f"filters must be a JSON object, got {type(filters).__name__}")


def _visible_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Drop bookkeeping fields an agent should not see or filter on."""
    return {
        k: v
        for k, v in metadata.items()
        if k not in INTERNAL_METADATA_KEYS and not k.startswith("_") and v is not None
    }


def format_documents(documents: List[Any]) -> str:
    """
    Render retrieved chunks as numbered, attributed text.

    Args:
        documents: Retrieved documents, best first

    Returns:
        One block per chunk with its source and metadata, or a clear statement
        that nothing was found. An empty result is reported explicitly so the
        agent does not read silence as "the tool broke" and retry forever.
    """
    if not documents:
        return (
            "No matching documents found. The collection may not contain this "
            "topic, or the filters may be too narrow. Call get_filter_context "
            "to see what is actually stored before retrying."
        )

    blocks = []
    for position, document in enumerate(documents, start=1):
        metadata = _visible_metadata(dict(document.metadata))
        source = metadata.pop("source", "unknown")

        text = document.page_content
        if len(text) > MAX_SNIPPET_CHARS:
            text = text[:MAX_SNIPPET_CHARS].rstrip() + "\n[truncated]"

        details = ", ".join(f"{k}={v}" for k, v in sorted(metadata.items()))
        header = f"[{position}] source: {source}"
        if details:
            header += f" ({details})"

        blocks.append(f"{header}\n{text}")

    return "\n\n---\n\n".join(blocks)


def search_documents(
    rag: Any,
    query: str,
    top_k: int = 5,
    filters: Optional[Any] = None,
) -> str:
    """
    Search the collection and return the matching chunks.

    Args:
        rag: A RAGWire instance
        query: What to search for
        top_k: How many chunks to return
        filters: Optional metadata filters, as a mapping or JSON string

    Returns:
        The matching chunks with their sources, as text
    """
    parsed = _parse_filters(filters)
    documents = rag.retrieve(query, top_k=top_k, filters=parsed)
    logger.info(f"MCP search returned {len(documents)} chunks for: {query[:60]}")
    return format_documents(documents)


def answer_question(
    rag: Any,
    question: str,
    top_k: int = 5,
    filters: Optional[Any] = None,
) -> str:
    """
    Answer a question from the collection, with sources.

    Args:
        rag: A RAGWire instance
        question: The question to answer
        top_k: How many chunks to ground the answer in
        filters: Optional metadata filters, as a mapping or JSON string

    Returns:
        The answer with its cited sources, or a statement that the collection
        does not contain the answer
    """
    parsed = _parse_filters(filters)
    answer = rag.query(question, top_k=top_k, filters=parsed)

    if answer.refused:
        return (
            f"{answer.text}\n\n"
            f"Do not answer this from your own knowledge. Tell the user the "
            f"documents do not cover it."
        )

    return answer.formatted()


def get_filter_context(rag: Any, query: str = "") -> str:
    """
    Describe what is in the collection and which filters would apply.

    This is the tool an agent should call first. It reports the metadata
    fields that exist and the values actually stored in them, so the agent
    filters on real values rather than guessing at spellings.

    Args:
        rag: A RAGWire instance
        query: Optional query, used to suggest filters for it

    Returns:
        A description of the available fields and their stored values
    """
    return rag.get_filter_context(query)


def collection_stats(rag: Any) -> str:
    """
    Report what the collection holds.

    Args:
        rag: A RAGWire instance

    Returns:
        Collection name, chunk count and configured retrieval settings
    """
    stats = rag.get_stats()
    lines = [f"{key}: {value}" for key, value in sorted(stats.items())]
    return "\n".join(lines) if lines else "No collection statistics available."
