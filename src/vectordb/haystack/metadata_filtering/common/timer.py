"""Timing utilities for measuring pipeline performance.

Provides Timer context manager for lightweight timing measurements.
"""

import time
from typing import Any


__all__ = ["Timer"]


class Timer:
    """Context manager for measuring elapsed time with perf_counter precision.

    Captures start and end times using time.perf_counter() for lightweight
    timing without instrumentation overhead.

    Attributes:
        start_time: Timestamp when context entered (0.0 if not started).
        end_time: Timestamp when context exited (0.0 if not exited).
    """

    def __init__(self) -> None:
        """Initialize Timer with zero start/end times."""
        self.start_time: float = 0.0
        self.end_time: float = 0.0

    def __enter__(self) -> "Timer":
        """Enter context manager and start timer.

        Returns:
            Self for use in 'with' statement.
        """
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager and stop timer.

        Args:
            exc_type: Exception type if raised.
            exc_val: Exception value if raised.
            exc_tb: Exception traceback if raised.
        """
        self.end_time = time.perf_counter()

    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds.

        Returns:
            Elapsed time in milliseconds (end_time - start_time) * 1000.
            Returns 0.0 if timer not started/stopped.
        """
        if self.start_time == 0.0 or self.end_time == 0.0:
            return 0.0
        return (self.end_time - self.start_time) * 1000.0
