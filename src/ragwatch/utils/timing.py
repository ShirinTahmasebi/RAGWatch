"""Timing utilities for measuring pipeline latency."""

import time


class Timer:
    """A simple context-manager timer that records elapsed time in milliseconds."""

    def __init__(self) -> None:
        self._start: float = 0.0
        self._end: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: object) -> None:
        self._end = time.perf_counter()

    @property
    def elapsed_ms(self) -> float:
        """Elapsed time in milliseconds."""
        return (self._end - self._start) * 1000.0
