"""
An MCP server exposing a RAGWire collection to agents.

Model Context Protocol is how Claude Desktop, Claude Code, Cursor and similar
clients discover and call tools. Running this turns an ingested collection
into something those clients can search directly, with no Python written by
the person using it.

The tool set is deliberately small and ordered around one idea: an agent
should find out what is in the collection before it guesses at filters.
``get_filter_context`` reports the fields and the values actually stored, and
``search_documents`` applies whatever the agent decides on.

The ``mcp`` package is an optional dependency. Install it with
``pip install ragwire[mcp]``.
"""

import logging
from typing import Any, Optional

from . import tools

logger = logging.getLogger(__name__)

SERVER_INSTRUCTIONS = """This server searches a private document collection that the user has ingested with RAGWire.

Use it whenever the user asks about their documents, filings, reports or any topic the collection might cover. Do not answer from your own knowledge when this collection could hold the answer.

Recommended order:
1. Call get_filter_context first if you are unsure what the collection contains. It reports the metadata fields and the values actually stored in them.
2. Call search_documents with filters drawn from those real values, never from guessed spellings.
3. Call answer_question instead when the user wants a direct answer rather than the underlying chunks. It returns an answer with citations, and tells you when the documents do not cover the question.

If a search returns nothing, widen or drop the filters before concluding the collection lacks the information."""


def build_server(rag: Any, name: str = "ragwire") -> Any:
    """
    Build an MCP server bound to a RAGWire instance.

    Args:
        rag: A RAGWire instance to expose
        name: Server name shown to the client

    Returns:
        A configured FastMCP server, ready to ``run()``

    Raises:
        ImportError: If the mcp package is not installed

    Example:
        >>> from ragwire import RAGWire                    # doctest: +SKIP
        >>> server = build_server(RAGWire("config.yaml"))  # doctest: +SKIP
        >>> server.run()                                   # doctest: +SKIP
    """
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise ImportError(
            "The MCP server requires the mcp package. "
            "Install it with: pip install ragwire[mcp]"
        ) from exc

    server = FastMCP(name, instructions=SERVER_INSTRUCTIONS)

    @server.tool()
    def get_filter_context(query: str = "") -> str:
        """Describe what the document collection contains: which metadata fields exist and which values are actually stored in them. Call this before searching if you are unsure what is available, so filters use real values rather than guessed ones."""
        return tools.get_filter_context(rag, query)

    @server.tool()
    def search_documents(
        query: str, top_k: int = 5, filters: Optional[dict] = None
    ) -> str:
        """Search the document collection and return the matching passages with their sources. Pass filters (for example {"company_name": "apple", "fiscal_year": 2025}) only with values confirmed via get_filter_context. Omit filters for a broad search."""
        return tools.search_documents(rag, query, top_k=top_k, filters=filters)

    @server.tool()
    def answer_question(
        question: str, top_k: int = 5, filters: Optional[dict] = None
    ) -> str:
        """Answer a question using only the document collection, returning the answer with numbered citations. Use this when the user wants an answer rather than raw passages. It reports plainly when the documents do not cover the question, and you must not fill that gap from your own knowledge."""
        return tools.answer_question(rag, question, top_k=top_k, filters=filters)

    @server.tool()
    def collection_stats() -> str:
        """Report the collection name, how many chunks it holds and its vector size. Useful for checking that documents were actually ingested."""
        return tools.collection_stats(rag)

    logger.info(f"MCP server '{name}' built with 4 tools")
    return server


def serve(config_path: str = "config.yaml", name: str = "ragwire") -> None:
    """
    Build and run an MCP server over stdio.

    stdio is the transport every desktop MCP client speaks, and it means the
    server writes nothing to stdout except protocol traffic. Logging is
    redirected to stderr for that reason: a stray print would corrupt the
    stream and the client would drop the connection.

    Args:
        config_path: Path to the RAGWire config file
        name: Server name shown to the client

    Raises:
        ImportError: If the mcp package is not installed
        FileNotFoundError: If the config file does not exist
    """
    import sys

    from ..core.pipeline import RAGWire

    # Anything on stdout is protocol traffic, so logs go to stderr.
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)

    rag = RAGWire(config_path)
    server = build_server(rag, name=name)

    logger.info(f"Serving collection over MCP (config={config_path})")
    server.run()
