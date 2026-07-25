# Switch from OpenAI to Local Ollama

Swap your embedding and LLM to run fully offline with no API costs. Only `config.yaml` changes; your Python code stays the same.

```yaml
embeddings:
  provider: "ollama"
  model: "nomic-embed-text"
  base_url: "http://localhost:11434"

llm:
  provider: "ollama"
  model: "qwen3.5:9b"
  base_url: "http://localhost:11434"
  num_ctx: 16384
```

## Switch to OpenRouter (one key, many models)

[OpenRouter](https://openrouter.ai) gives you a single API key for both the LLM and embeddings, with free-tier models available. Requires `pip install "ragwire[openrouter]"` (Python ≥ 3.10).

```yaml
embeddings:
  provider: "openrouter"
  model: "nvidia/llama-nemotron-embed-vl-1b-v2:free"
  api_key: "${OPENROUTER_API_KEY}"

llm:
  provider: "openrouter"
  model: "poolside/laguna-m.1:free"
  api_key: "${OPENROUTER_API_KEY}"
```

See the [OpenRouter provider guide](../openrouter.md) for details.

!!! warning "Recreate the collection when switching embedding models"
    Different embedding models produce different vector dimensions. If you already have a collection, set `force_recreate: true` once, run ingestion, then set it back to `false`.

For all available providers and their config values, see the [Providers](../ollama.md) section.
