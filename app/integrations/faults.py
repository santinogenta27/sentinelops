"""Deterministic fault injection for tools (used by tests, the eval suite and the demo)."""

from __future__ import annotations

import threading
from collections import defaultdict, deque
from typing import Literal

FaultKind = Literal["timeout", "error", "invalid_args"]


class ToolError(Exception):
    """Base error raised by tools. ``retryable`` drives the self-correction loop."""

    def __init__(self, message: str, *, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable


class ToolTimeoutError(ToolError):
    pass


class FaultInjector:
    """Queue of faults per tool name; each call to :meth:`check` consumes one."""

    def __init__(self) -> None:
        self._queues: dict[str, deque[FaultKind]] = defaultdict(deque)
        self._lock = threading.Lock()

    def fail_next(self, tool: str, times: int = 1, kind: FaultKind = "timeout") -> None:
        with self._lock:
            self._queues[tool].extend([kind] * times)

    def clear(self) -> None:
        with self._lock:
            self._queues.clear()

    def check(self, tool: str) -> None:
        with self._lock:
            queue = self._queues.get(tool)
            kind = queue.popleft() if queue else None
        if kind == "timeout":
            raise ToolTimeoutError(f"{tool}: upstream timed out after 5000ms (simulated)")
        if kind == "error":
            raise ToolError(f"{tool}: upstream returned HTTP 503 Service Unavailable (simulated)")
        if kind == "invalid_args":
            raise ToolError(f"{tool}: 422 invalid parameters (simulated)", retryable=True)


fault_injector = FaultInjector()
