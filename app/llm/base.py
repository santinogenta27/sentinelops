"""Provider-agnostic LLM client with structured output, self-repair, rate limiting and a circuit breaker."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from collections import deque
from enum import StrEnum
from typing import TypeVar

from pydantic import BaseModel, ValidationError

logger = logging.getLogger("sentinelops.llm")

T = TypeVar("T", bound=BaseModel)


class LLMTask(StrEnum):
    TRIAGE = "triage"
    REPAIR_TOOL_ARGS = "repair_tool_args"
    INVESTIGATION_SUMMARY = "investigation_summary"
    DRAFT_RESPONSE = "draft_response"
    JUDGE = "judge"


class LLMError(Exception):
    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


class LLMUnavailableError(LLMError):
    """Raised when the circuit is open or the provider is permanently unusable (e.g. bad key)."""

    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=False)


class RateLimiter:
    """Async sliding-window limiter (requests per minute) shared by all calls of a client."""

    def __init__(self, rpm: int) -> None:
        self.rpm = rpm
        self._calls: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            while True:
                now = time.monotonic()
                while self._calls and now - self._calls[0] > 60.0:
                    self._calls.popleft()
                if len(self._calls) < self.rpm:
                    self._calls.append(now)
                    return
                await asyncio.sleep(60.0 - (now - self._calls[0]) + 0.05)


class CircuitBreaker:
    """Opens after ``threshold`` consecutive failures; permanently opens on auth errors."""

    def __init__(self, threshold: int = 3, cooldown_s: float = 60.0) -> None:
        self.threshold = threshold
        self.cooldown_s = cooldown_s
        self._failures = 0
        self._opened_at: float | None = None
        self._permanent_reason: str | None = None

    @property
    def state(self) -> str:
        if self._permanent_reason:
            return "open:permanent"
        if self._opened_at is None:
            return "closed"
        return "open" if time.monotonic() - self._opened_at < self.cooldown_s else "half-open"

    @property
    def reason(self) -> str | None:
        return self._permanent_reason

    def allow(self) -> bool:
        return self.state in ("closed", "half-open")

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None

    def record_failure(self, *, permanent_reason: str | None = None) -> None:
        if permanent_reason:
            self._permanent_reason = permanent_reason
            return
        self._failures += 1
        if self._failures >= self.threshold:
            self._opened_at = time.monotonic()


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def extract_json(text: str) -> object:
    cleaned = _FENCE_RE.sub("", text.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise
        return json.loads(cleaned[start : end + 1])


class LLMClient(ABC):
    """Template-method base: subclasses implement :meth:`_complete` only."""

    provider: str = "abstract"

    def __init__(self, *, model: str, rpm: int, max_retries: int, timeout_s: float) -> None:
        self.model = model
        self.max_retries = max_retries
        self.timeout_s = timeout_s
        self.limiter = RateLimiter(rpm)
        self.breaker = CircuitBreaker()
        self.calls_made = 0

    @abstractmethod
    async def _complete(self, *, system: str, prompt: str, json_mode: bool) -> str: ...

    async def _call(self, *, task: LLMTask, system: str, prompt: str, json_mode: bool) -> str:
        if not self.breaker.allow():
            raise LLMUnavailableError(f"{self.provider} circuit open ({self.breaker.reason or 'cooling down'})")
        delay = 1.5
        last: Exception | None = None
        for attempt in range(1, self.max_retries + 2):
            await self.limiter.acquire()
            try:
                self.calls_made += 1
                text = await asyncio.wait_for(
                    self._complete(system=system, prompt=prompt, json_mode=json_mode), timeout=self.timeout_s
                )
                self.breaker.record_success()
                return text
            except TimeoutError as exc:
                last = LLMError(f"timeout after {self.timeout_s}s", retryable=True)
                logger.warning("llm %s timeout (attempt %d): %s", task, attempt, exc)
            except LLMError as exc:
                last = exc
                if not exc.retryable:
                    if isinstance(exc, LLMUnavailableError):
                        self.breaker.record_failure(permanent_reason=str(exc))
                    else:
                        self.breaker.record_failure()
                    raise
                logger.warning("llm %s retryable error (attempt %d): %s", task, attempt, exc)
            if attempt <= self.max_retries:
                await asyncio.sleep(delay)
                delay = min(delay * 2, 20.0)
        self.breaker.record_failure()
        raise last if last else LLMError("unknown LLM failure", retryable=False)

    async def generate_text(self, *, task: LLMTask, system: str, prompt: str) -> str:
        return (await self._call(task=task, system=system, prompt=prompt, json_mode=False)).strip()

    async def generate_structured(
        self, *, task: LLMTask, system: str, prompt: str, schema: type[T], repair_attempts: int = 2
    ) -> T:
        """JSON-mode generation validated by Pydantic; validation errors are fed back for self-repair."""
        schema_json = json.dumps(schema.model_json_schema(), indent=None)
        full_prompt = f"{prompt}\n\nRespond ONLY with a JSON object that validates against this JSON Schema:\n{schema_json}"
        current = full_prompt
        last_error: Exception | None = None
        for _ in range(repair_attempts + 1):
            raw = await self._call(task=task, system=system, prompt=current, json_mode=True)
            try:
                return schema.model_validate(extract_json(raw))
            except (ValidationError, json.JSONDecodeError, ValueError) as exc:
                last_error = exc
                current = (
                    f"{full_prompt}\n\nYour previous answer was INVALID.\nPrevious answer:\n{raw[:2000]}\n"
                    f"Validation errors:\n{str(exc)[:1500]}\nReturn a corrected JSON object only."
                )
        raise LLMError(f"structured output failed validation: {last_error}", retryable=False)

    def status(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "model": self.model,
            "circuit": self.breaker.state,
            "circuit_reason": self.breaker.reason,
            "calls_made": self.calls_made,
        }
