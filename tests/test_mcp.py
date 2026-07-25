"""
Tests for the MCP tool layer and the CLI parser.

The tool functions are plain functions over a RAGWire-shaped object, so they
are tested without the optional mcp package. The FastMCP registration in
server.py is a thin wrapper over these and is not exercised here, since it
cannot run without that dependency installed.
"""

import pytest
from langchain_core.documents import Document

from ragwire.cli import build_parser, main
from ragwire.mcp import tools


class _StubRAG:
    """Records what it was asked for and returns canned results."""

    def __init__(self, documents=None, answer=None, stats=None, context="context text"):
        self._documents = documents if documents is not None else []
        self._answer = answer
        self._stats = stats or {}
        self._context = context
        self.calls = []

    def retrieve(self, query, top_k=5, filters=None, rerank=None):
        self.calls.append(("retrieve", query, top_k, filters))
        return self._documents

    def query(self, question, top_k=5, filters=None, rerank=None):
        self.calls.append(("query", question, top_k, filters))
        return self._answer

    def get_filter_context(self, query):
        self.calls.append(("filter_context", query))
        return self._context

    def get_stats(self):
        return self._stats


class _StubAnswer:
    def __init__(self, text, refused=False):
        self.text = text
        self.refused = refused

    def formatted(self):
        return f"{self.text}\n\nSources:\n  [1] a.pdf"


def _docs(*sources, text="chunk body"):
    return [
        Document(page_content=f"{text} from {s}", metadata={"source": s})
        for s in sources
    ]


# --------------------------------------------------------------------------- #
# Filter parsing
# --------------------------------------------------------------------------- #

def test_a_mapping_of_filters_passes_through():
    assert tools._parse_filters({"company_name": "apple"}) == {"company_name": "apple"}


def test_a_json_string_of_filters_is_parsed():
    # Models routinely send a JSON string where the schema asked for an object.
    assert tools._parse_filters('{"fiscal_year": 2025}') == {"fiscal_year": 2025}


def test_absent_filters_become_none():
    assert tools._parse_filters(None) is None
    assert tools._parse_filters("") is None
    assert tools._parse_filters({}) is None


def test_unparseable_filters_raise_a_message_the_agent_can_act_on():
    with pytest.raises(ValueError) as excinfo:
        tools._parse_filters("company_name=apple")

    assert "JSON object" in str(excinfo.value)


def test_a_json_list_is_not_a_valid_filter_object():
    with pytest.raises(ValueError):
        tools._parse_filters("[1, 2]")


# --------------------------------------------------------------------------- #
# Document formatting
# --------------------------------------------------------------------------- #

def test_results_are_numbered_and_attributed():
    output = tools.format_documents(_docs("a.pdf", "b.pdf"))

    assert "[1] source: a.pdf" in output
    assert "[2] source: b.pdf" in output


def test_an_empty_result_says_so_and_suggests_what_to_do():
    output = tools.format_documents([])

    # Silence reads as a broken tool, and the agent retries forever.
    assert "No matching documents" in output
    assert "get_filter_context" in output


def test_bookkeeping_metadata_is_hidden_from_the_agent():
    doc = Document(
        page_content="text",
        metadata={
            "source": "a.pdf",
            "company_name": "apple",
            "content_hash": "deadbeef",
            "total_chunks": 12,
            "_id": "xyz",
        },
    )

    output = tools.format_documents([doc])

    assert "company_name=apple" in output
    assert "content_hash" not in output
    assert "total_chunks" not in output
    assert "_id" not in output


def test_empty_metadata_values_are_not_shown():
    doc = Document(page_content="t", metadata={"source": "a.pdf", "fiscal_year": None})

    assert "fiscal_year" not in tools.format_documents([doc])


def test_long_chunks_are_truncated():
    doc = Document(page_content="x" * 5000, metadata={"source": "a.pdf"})

    output = tools.format_documents([doc])

    assert "[truncated]" in output
    assert len(output) < 2500


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #

def test_search_documents_forwards_the_query_and_filters():
    rag = _StubRAG(documents=_docs("a.pdf"))

    output = tools.search_documents(rag, "revenue", top_k=3, filters='{"fiscal_year": 2025}')

    assert rag.calls[0] == ("retrieve", "revenue", 3, {"fiscal_year": 2025})
    assert "a.pdf" in output


def test_answer_question_returns_the_answer_with_its_sources():
    rag = _StubRAG(answer=_StubAnswer("Net income was $93.7bn [1]."))

    output = tools.answer_question(rag, "net income?")

    assert "Net income was $93.7bn [1]." in output
    assert "Sources:" in output


def test_a_refusal_tells_the_agent_not_to_fill_the_gap_itself():
    rag = _StubRAG(answer=_StubAnswer("The documents do not cover this.", refused=True))

    output = tools.answer_question(rag, "capital of France?")

    # Without this the agent answers from its own knowledge and the collection
    # boundary silently disappears.
    assert "Do not answer this from your own knowledge" in output


def test_get_filter_context_passes_the_query_through():
    rag = _StubRAG(context="fields: company_name")

    assert tools.get_filter_context(rag, "apple revenue") == "fields: company_name"
    assert rag.calls[0] == ("filter_context", "apple revenue")


def test_collection_stats_renders_key_values():
    rag = _StubRAG(stats={"collection_name": "financial_docs", "total_documents": 42})

    output = tools.collection_stats(rag)

    assert "collection_name: financial_docs" in output
    assert "total_documents: 42" in output


def test_collection_stats_handles_an_empty_response():
    assert "No collection statistics" in tools.collection_stats(_StubRAG())


# --------------------------------------------------------------------------- #
# Server construction
# --------------------------------------------------------------------------- #

def test_the_server_registers_the_four_tools():
    pytest.importorskip("mcp", reason="MCP server requires: pip install ragwire[mcp]")

    import asyncio

    from ragwire.mcp.server import build_server

    server = build_server(_StubRAG(documents=_docs("a.pdf")))
    names = {tool.name for tool in asyncio.run(server.list_tools())}

    assert names == {
        "get_filter_context",
        "search_documents",
        "answer_question",
        "collection_stats",
    }


def test_every_tool_carries_a_description_the_agent_can_route_on():
    pytest.importorskip("mcp", reason="MCP server requires: pip install ragwire[mcp]")

    import asyncio

    from ragwire.mcp.server import build_server

    for tool in asyncio.run(build_server(_StubRAG()).list_tools()):
        # An undescribed tool is one the agent will call at random, or never.
        assert tool.description and len(tool.description) > 40


def test_a_registered_tool_reaches_the_underlying_pipeline():
    pytest.importorskip("mcp", reason="MCP server requires: pip install ragwire[mcp]")

    import asyncio

    from ragwire.mcp.server import build_server

    rag = _StubRAG(documents=_docs("a.pdf"))
    server = build_server(rag)

    asyncio.run(server.call_tool("search_documents", {"query": "revenue", "top_k": 2}))

    assert rag.calls[0] == ("retrieve", "revenue", 2, None)


def test_the_tool_functions_import_without_the_mcp_package():
    # ragwire.mcp must stay importable on a base install, since the tool
    # functions need nothing beyond it. Attribute access is what pulls the
    # optional dependency in.
    from ragwire.mcp import search_documents  # noqa: F401


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def test_mcp_serve_parses_with_defaults():
    args = build_parser().parse_args(["mcp", "serve"])

    assert args.command == "mcp"
    assert args.mcp_command == "serve"
    assert args.config == "config.yaml"


def test_mcp_serve_accepts_a_config_and_name():
    args = build_parser().parse_args(
        ["mcp", "serve", "--config", "other.yaml", "--name", "filings"]
    )

    assert args.config == "other.yaml"
    assert args.name == "filings"


def test_ingest_parses_a_path_and_recursive_flag():
    args = build_parser().parse_args(["ingest", "./docs", "--recursive"])

    assert args.path == "./docs"
    assert args.recursive is True


def test_eval_parses_the_golden_set_and_options():
    args = build_parser().parse_args(
        ["eval", "golden.yaml", "--top-k", "10", "--compare-rerank"]
    )

    assert args.golden == "golden.yaml"
    assert args.top_k == 10
    assert args.compare_rerank is True


def test_version_prints_and_exits_zero(capsys):
    assert main(["--version"]) == 0
    assert "ragwire" in capsys.readouterr().out


def test_no_command_prints_help_and_fails():
    assert main([]) == 1


def test_mcp_without_a_subcommand_is_a_usage_error(capsys):
    assert main(["mcp"]) == 2
    assert "ragwire mcp serve" in capsys.readouterr().err
