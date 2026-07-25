"""
Retry helper for transient network failures.

Every remote call in a RAG pipeline (LLM, embedding provider, vector store)
can fail for reasons that resolve on their own: a rate limit, a restarting
container, a dropped connection. Without a retry those become permanent
per-document failures, which is how a single blip used to cost a whole batch.
"""

import logging
import time
from typing import Any, Callable, Iterable, Optional, Tuple, Type

logger = logging.getLogger(__name__)

#: Exceptions that never resolve by trying again. These signal a bug or a
#: misconfiguration, so retrying just delays the error the caller needs to see.
NON_RETRYABLE: Tuple[Type[BaseException], ...] = (
    TypeError,
    AttributeError,
    ImportError,
    NameError,
    SyntaxError,
    KeyboardInterrupt,
    SystemExit,
)


def retry_call(
    func: Callable[[], Any],
    attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    description: str = "operation",
    non_retryable: Optional[Iterable[Type[BaseException]]] = None,
) -> Any:
    """
    Call ``func``, retrying transient failures with exponential backoff.

    Args:
        func: Zero-argument callable to invoke
        attempts: Total attempts including the first (1 disables retrying)
        base_delay: Seconds to wait after the first failure; doubles each time
        max_delay: Upper bound on the backoff delay
        description: Used in log messages, e.g. "write batch 2/7"
        non_retryable: Exception types to re-raise immediately.
                       Defaults to NON_RETRYABLE.

    Returns:
        Whatever ``func`` returns

    Raises:
        The last exception, if every attempt failed

    Example:
        >>> retry_call(lambda: store.add_documents(batch), attempts=3)
    """
    blocked = tuple(non_retryable) if non_retryable is not None else NON_RETRYABLE
    total = max(1, attempts)

    for attempt in range(1, total + 1):
        try:
            return func()
        except blocked:
            raise
        except Exception as e:
            if attempt >= total:
                logger.error(
                    f"{description} failed after {total} attempt(s): {e}"
                )
                raise
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            logger.warning(
                f"{description} failed (attempt {attempt}/{total}): {e}. "
                f"Retrying in {delay:.0f}s"
            )
            time.sleep(delay)
