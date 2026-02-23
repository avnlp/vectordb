"""Timing utilities for namespace pipelines."""

from __future__ import annotations

import time
from typing import Any


class Timer:
    """Context manager for timing operations.

    Example:
        >>> with Timer() as t:
        ...     # do some work
        >>> print(f"Took {t.elapsed_ms:.2f}ms")
    """

    def __init__(self) -> None:
        """Initialize timer."""
        self._start: float = 0.0
        self._end: float = 0.0

    def __enter__(self) -> "Timer":
        """Start the timer."""
        self._start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Stop the timer."""
        self._end = time.perf_counter()

    @property
    def elapsed_ms(self) -> float:
        """Return elapsed time in milliseconds."""
        return (self._end - self._start) * 1000
