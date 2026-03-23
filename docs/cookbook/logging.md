# Logging Setup

## Write Logs to a File

Useful in production or scheduled jobs where you want a persistent audit trail.

```yaml
logging:
  level: "INFO"
  colored: false
  console_output: true
  log_file: "logs/rag.log"
```

Or in code:

```python
from ragwire import setup_logging

logger = setup_logging(log_level="INFO", log_file="logs/rag.log")
```

## Colored Output for Development

```python
from ragwire import setup_colored_logging

logger = setup_colored_logging(log_level="DEBUG")
```

Or via config:

```yaml
logging:
  level: "DEBUG"
  colored: true
```

## Log Levels

| Level | Use When |
|---|---|
| `DEBUG` | Development — shows every step (loading, chunking, embedding, storing) |
| `INFO` | Production — shows ingestion stats, retrieval queries, warnings |
| `WARNING` | Minimal — only problems and skipped files |
| `ERROR` | Errors only |

## Silence Third-Party Noise

Some providers (HuggingFace, fastembed) emit verbose logs. Silence them while keeping RAGWire logs:

```python
import logging

logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("fastembed").setLevel(logging.WARNING)

from ragwire import setup_logging
logger = setup_logging(log_level="INFO")
```
