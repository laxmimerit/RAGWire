# LangSmith Tracing

RAGWire uses LangChain internally for LLM calls (metadata extraction, filter extraction). Because of this, **LangSmith tracing works out of the box** — no code changes needed. Just set four environment variables and every LLM call RAGWire makes will appear in your LangSmith dashboard.

## Enable Tracing

Add the following to your `.env` file in the project root:

```env
LANGSMITH_API_KEY=your_api_key_here
LANGSMITH_PROJECT=RAGWire
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_TRACING_V2=true
```

Get your API key from [smith.langchain.com](https://smith.langchain.com).

RAGWire loads `.env` automatically at startup via `python-dotenv` — no additional setup required.

## What Gets Traced

Every LLM call RAGWire makes is captured automatically:

| Operation | What You See in LangSmith |
|---|---|
| `ingest_documents()` | Metadata extraction prompt + LLM response for each document |
| `retrieve()` (auto-filter) | Filter extraction prompt + LLM response for each query |
| `extract_metadata()` | Full prompt with grounding + extracted JSON |

## What to Expect in the Dashboard

Each trace shows:
- The full prompt sent to the LLM (including grounding values if the collection has data)
- The raw LLM response before JSON parsing
- Latency per call
- Token usage (if supported by the provider)

This is especially useful for debugging why the LLM extracted `"apple"` instead of `"apple inc."` — you can inspect the exact prompt and grounding that was passed.

## Disable Tracing

Remove the variables or set:

```env
LANGCHAIN_TRACING_V2=false
```
