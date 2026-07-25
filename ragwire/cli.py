"""
Command line interface for RAGWire.

Three things are worth doing without writing a script: serving a collection to
an agent, ingesting a folder, and checking whether retrieval actually works.
Everything else stays in the Python API.

.. code-block:: bash

    ragwire mcp serve --config config.yaml
    ragwire ingest ./documents --config config.yaml
    ragwire eval golden.yaml --config config.yaml --compare-rerank
"""

import argparse
import logging
import sys
from typing import List, Optional

logger = logging.getLogger(__name__)


def _add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the RAGWire config file (default: config.yaml)",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser. Split out so tests can exercise it directly."""
    parser = argparse.ArgumentParser(
        prog="ragwire",
        description="RAGWire: production-grade RAG toolkit for document ingestion and retrieval.",
    )
    parser.add_argument("--version", action="store_true", help="Print the installed version and exit")

    subcommands = parser.add_subparsers(dest="command")

    # ----------------------------------------------------------------- mcp
    mcp_parser = subcommands.add_parser(
        "mcp", help="Expose a collection to MCP clients such as Claude Desktop"
    )
    mcp_sub = mcp_parser.add_subparsers(dest="mcp_command")
    serve_parser = mcp_sub.add_parser("serve", help="Run the MCP server over stdio")
    _add_config_arg(serve_parser)
    serve_parser.add_argument(
        "--name",
        default="ragwire",
        help="Server name shown to the client (default: ragwire)",
    )

    # -------------------------------------------------------------- ingest
    ingest_parser = subcommands.add_parser(
        "ingest", help="Ingest files or a directory into the collection"
    )
    ingest_parser.add_argument("path", help="File or directory to ingest")
    _add_config_arg(ingest_parser)
    ingest_parser.add_argument(
        "--recursive", action="store_true", help="Scan subdirectories too"
    )

    # ---------------------------------------------------------------- eval
    eval_parser = subcommands.add_parser(
        "eval", help="Score retrieval against a golden set"
    )
    eval_parser.add_argument("golden", help="Path to a golden set YAML or JSON file")
    _add_config_arg(eval_parser)
    eval_parser.add_argument("--top-k", type=int, default=5, help="Cutoff for every metric (default: 5)")
    eval_parser.add_argument(
        "--compare-rerank",
        action="store_true",
        help="Run with and without reranking and print a comparison table",
    )

    return parser


def _cmd_version() -> int:
    from . import __version__

    print(f"ragwire {__version__}")
    return 0


def _cmd_mcp_serve(args: argparse.Namespace) -> int:
    from .mcp.server import serve

    serve(config_path=args.config, name=args.name)
    return 0


def _cmd_ingest(args: argparse.Namespace) -> int:
    from pathlib import Path

    from .core.pipeline import RAGWire

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    rag = RAGWire(args.config)
    target = Path(args.path)

    if target.is_dir():
        stats = rag.ingest_directory(str(target), recursive=args.recursive)
    else:
        stats = rag.ingest_documents([str(target)])

    print(
        f"ingested={stats['ingested']} skipped={stats['skipped']} "
        f"failed={len(stats.get('errors', []))}"
    )
    for error in stats.get("errors", []):
        print(f"  failed: {error.get('file')}: {error.get('error')}", file=sys.stderr)

    # A run where every file failed should not look like success to a shell.
    return 1 if stats.get("errors") and not stats["ingested"] else 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from .core.pipeline import RAGWire
    from .eval import GoldenSet, evaluate, sweep

    # Ingestion chatter would bury the table this command exists to print.
    logging.basicConfig(level=logging.WARNING)

    rag = RAGWire(args.config)
    golden = GoldenSet.from_file(args.golden)

    if args.compare_rerank:
        print(sweep(rag, golden, {
            "no rerank": {"rerank": False},
            "reranked": {"rerank": True},
        }, top_k=args.top_k))
    else:
        print(evaluate(rag, golden, top_k=args.top_k))

    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """
    Entry point for the ``ragwire`` command.

    Args:
        argv: Arguments to parse. Uses sys.argv when not given.

    Returns:
        A process exit code
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        return _cmd_version()

    if args.command == "mcp":
        if args.mcp_command != "serve":
            print("usage: ragwire mcp serve [--config CONFIG] [--name NAME]", file=sys.stderr)
            return 2
        return _cmd_mcp_serve(args)

    if args.command == "ingest":
        return _cmd_ingest(args)

    if args.command == "eval":
        return _cmd_eval(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
