"""
MCP server for RAGWire collections.

Exposes an ingested collection to Claude Desktop, Claude Code, Cursor and any
other Model Context Protocol client:

.. code-block:: bash

    pip install ragwire[mcp]
    ragwire mcp serve --config config.yaml

The tool implementations in :mod:`ragwire.mcp.tools` are plain functions and
carry no MCP dependency, so they can be reused wherever an agent needs to
reach a collection.
"""

from .tools import (
    answer_question,
    collection_stats,
    format_documents,
    get_filter_context,
    search_documents,
)

__all__ = [
    "search_documents",
    "answer_question",
    "get_filter_context",
    "collection_stats",
    "format_documents",
    "build_server",
    "serve",
]


def __getattr__(name):
    """
    Load the server lazily.

    Importing it eagerly would pull in the optional mcp package just to use
    the tool functions, which need nothing beyond the base install.
    """
    if name in ("build_server", "serve"):
        from . import server

        return getattr(server, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
