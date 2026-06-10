"""Async retry decorator for transient failure resilience."""

from __future__ import annotations

import asyncio
import functools
import logging
from collections.abc import Callable
from typing import Any, TypeVar

logger = logging.getLogger("hr_agent_system.retry")

F = TypeVar("F", bound=Callable[..., Any])


def async_retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> Callable[[F], F]:
    """Decorator factory that retries an async function on specified exceptions.

    Logs a WARNING on each failed attempt and re-raises the last exception
    after max_attempts are exhausted.
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: BaseException | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    logger.warning(
                        "Retry attempt %d/%d for %s: %s",
                        attempt,
                        max_attempts,
                        func.__name__,
                        exc,
                    )
                    if attempt < max_attempts:
                        await asyncio.sleep(delay)
            raise last_exc  # type: ignore[misc]

        return wrapper  # type: ignore[return-value]

    return decorator
