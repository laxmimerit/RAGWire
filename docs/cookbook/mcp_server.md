# MCP Server

Turn an ingested collection into a tool that Claude Desktop, Claude Code and Cursor can call directly. One command, no Python written by whoever uses it.

```bash
pip install ragwire[mcp]
ragwire mcp serve --config config.yaml
```

## What your agent gets

Four tools, ordered around one idea: an agent should find out what is in the collection before guessing at filters.

| Tool | What it does |
|---|---|
| `get_filter_context` | Reports the metadata fields and the values actually stored in them |
| `search_documents` | Returns matching passages with their sources |
| `answer_question` | Returns a cited answer, and says plainly when the documents do not cover it |
| `collection_stats` | Collection name, chunk count, vector size |

`get_filter_context` is the one that makes the rest work. Without it an agent filtering on company name has to guess whether you stored `"Apple"`, `"apple"` or `"Apple Inc."`, and a wrong guess returns nothing, which reads to the agent like an empty collection. With it, the agent sees the real values first.

## Claude Desktop

Edit the config file:

- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "ragwire": {
      "command": "ragwire",
      "args": ["mcp", "serve", "--config", "C:/path/to/config.yaml"]
    }
  }
}
```

Use an absolute path to `config.yaml`. The server is launched by Claude Desktop from a working directory you do not control, so a relative path will not resolve. Paths written inside the config are fine either way: `metadata.config_file` resolves against the config file's own directory, so it can stay relative.

Restart Claude Desktop. The tools appear under the connector icon.

## Claude Code

```bash
claude mcp add ragwire -- ragwire mcp serve --config /absolute/path/to/config.yaml
```

## Cursor

In `.cursor/mcp.json` in your project, or the global equivalent:

```json
{
  "mcpServers": {
    "ragwire": {
      "command": "ragwire",
      "args": ["mcp", "serve", "--config", "/absolute/path/to/config.yaml"]
    }
  }
}
```

## Serving several collections

Each collection is its own server entry. Give each one a name, because the name is most of what the agent uses to decide which to call:

```json
{
  "mcpServers": {
    "sec-filings": {
      "command": "ragwire",
      "args": ["mcp", "serve", "--config", "/work/filings.yaml", "--name", "sec-filings"]
    },
    "internal-wiki": {
      "command": "ragwire",
      "args": ["mcp", "serve", "--config", "/work/wiki.yaml", "--name", "internal-wiki"]
    }
  }
}
```

## Making answers better

Everything else in RAGWire applies to the MCP server, because it is the same pipeline:

- Turn on [reranking](reranking.md) and `search_documents` returns better passages
- `answer_question` uses `rag.query()`, so it refuses rather than inventing when the documents come up short
- [Measure retrieval quality](evaluation.md) first, since a poor agent experience is almost always a retrieval problem rather than a prompting one

## Troubleshooting

**The server does not appear in the client.** Check that `ragwire` is on the PATH of the environment the client launches. If you installed into a virtualenv, point `command` at the absolute path of the executable inside it, for example `/home/you/venv/bin/ragwire` or `C:/venv/Scripts/ragwire.exe`.

**It appears but every call fails.** Run the exact command from your terminal. Configuration errors, a missing Qdrant, or an unreachable Ollama all surface immediately there and are invisible inside the client.

**Nothing is ever found.** Confirm the collection is populated:

```bash
python -c "from ragwire import RAGWire; print(RAGWire('config.yaml').get_stats())"
```

A `total_documents` of 0 means nothing was ingested, and no amount of agent prompting will fix it.

!!! note "stdout belongs to the protocol"
    The server speaks MCP over stdio, so anything printed to stdout corrupts the stream and the client drops the connection. RAGWire sends its logging to stderr for this reason. If you extend the server, do not `print()`.

## Using the tools without MCP

The tool implementations are plain functions with no MCP dependency, so they work anywhere an agent needs to reach a collection:

```python
from ragwire import RAGWire
from ragwire.mcp import search_documents, get_filter_context

rag = RAGWire("config.yaml")

context = get_filter_context(rag, "apple revenue")
results = search_documents(rag, "revenue", top_k=5, filters={"company_name": "apple"})
```

Both return agent-ready text. `filters` accepts a dict or a JSON string, since models routinely send a string where a schema asked for an object.

## See also

- [RAG Agent](../rag_agent.md) for building an agent in Python instead
- [Metadata-Aware Chatbot](filtered_chatbot.md) for the filter-context pattern in detail
