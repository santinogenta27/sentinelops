"""Provider-agnostic LLM client with structured output, self-repair, rate limiting and a circuit breaker."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass, fields, replace
from enum import StrEnum
from typing import Any, TypeVar

from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree
from pydantic import BaseModel, ValidationError

logger = logging.getLogger("sentinelops.llm")

T = TypeVar("T", bound=BaseModel)

# HTTP codes that mean "the provider/infrastructure failed", not "the model answered badly".
_INFRA_HTTP = {401, 403, 404, 408, 429}


class LLMTask(StrEnum):
    TRIAGE = "triage"
    REPAIR_TOOL_ARGS = "repair_tool_args"
    INVESTIGATION_SUMMARY = "investigation_summary"
    DRAFT_RESPONSE = "draft_response"
    JUDGE = "judge"


class LLMError(Exception):
    def __init__(self, message: str, *, retryable: bool, status_code: int | None = None) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code

    @property
    def is_infrastructure(self) -> bool:
        """True when the failure is the provider's (quota, outage, auth), not the model's answer."""
        code = self.status_code
        return self.retryable or (code is not None and (code in _INFRA_HTTP or code >= 500))

    @property
    def is_rate_limit(self) -> bool:
        return self.status_code == 429


class LLMUnavailableError(LLMError):
    """Raised when the circuit is open or the provider is permanently unusable (e.g. bad key)."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message, retryable=False, status_code=status_code)

    @property
    def is_infrastructure(self) -> bool:
        return True


@dataclass(slots=True)
class LLMUsage:
    """Bookkeeping only (never read by the agent): what the client did and what it cost."""

    calls: int = 0  # provider requests attempted (retries included)
    retries: int = 0
    rate_limited_attempts: int = 0  # HTTP 429 answers, whether or not a retry later succeeded
    surfaced_errors: int = 0  # LLMError raised to a node -> that node fell back to the deterministic path
    surfaced_infra_errors: int = 0  # ... of which caused by the provider (quota, outage, circuit open)
    surfaced_model_errors: int = 0  # ... of which caused by the model's answer (invalid structured output)
    input_tokens: int = 0
    output_tokens: int = 0  # includes thinking / reasoning tokens when the provider reports them
    calls_without_usage: int = 0  # successful calls whose provider response carried no token counts
    provider_seconds: float = 0.0
    limiter_wait_seconds: float = 0.0
    last_infra_error: str | None = None

    def snapshot(self) -> LLMUsage:
        return replace(self)

    def since(self, before: LLMUsage) -> LLMUsage:
        """Delta between this snapshot and an earlier one (last_infra_error kept only if it changed)."""
        out = LLMUsage()
        for f in fields(self):
            now, then = getattr(self, f.name), getattr(before, f.name)
            if isinstance(now, (int, float)) and not isinstance(now, bool):
                setattr(out, f.name, now - then)
        out.last_infra_error = self.last_infra_error if self.surfaced_infra_errors > before.surfaced_infra_errors else None
        return out

    def as_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


class RateLimiter:
    """Async sliding-window limiter (requests per minute) shared by all calls of a client."""

    def __init__(self, rpm: int) -> None:
        self.rpm = rpm
        self._calls: deque[float] = deque()
        self._lock = asyncio.Lock()

    def on_tokens(self, tokens: int) -> None:
        """Hook called with the real token count of each completed request (token-aware subclasses)."""

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
        self.usage = LLMUsage()

    @abstractmethod
    async def _complete(self, *, system: str, prompt: str, json_mode: bool) -> str: ...

    def _record_tokens(self, input_tokens: int | None, output_tokens: int | None) -> None:
        """Called by providers after each successful response. Missing counts are never treated as 0."""
        if input_tokens is None or output_tokens is None:
            self.usage.calls_without_usage += 1
            return
        self.usage.input_tokens += input_tokens
        self.usage.output_tokens += output_tokens
        self.limiter.on_tokens(input_tokens + output_tokens)
        holder = _CALL_TOKENS.get()
        if holder is not None:
            holder[0] += input_tokens
            holder[1] += output_tokens

    async def _call(self, *, task: LLMTask, system: str, prompt: str, json_mode: bool) -> str:
        if not tracing_is_enabled():
            return await self._call_impl(task=task, system=system, prompt=prompt, json_mode=json_mode)
        # LangSmith tracing (opt-in, see app/observability.py): one "llm" span per logical call, nested
        # under the LangGraph node that made it, with provider/model and token usage.
        traced = traceable(
            run_type="llm",
            name=f"llm.{task}",
            metadata={"ls_provider": self.provider, "ls_model_name": self.model, "task": str(task)},
        )(self._call_impl)
        result = await traced(task=task, system=system, prompt=prompt, json_mode=json_mode)
        return str(result)

    async def _call_impl(self, *, task: LLMTask, system: str, prompt: str, json_mode: bool) -> str:
        token = _CALL_TOKENS.set([0, 0])
        try:
            text = await self._call_with_retries(task=task, system=system, prompt=prompt, json_mode=json_mode)
            holder = _CALL_TOKENS.get()
            run = get_current_run_tree()
            if run is not None and holder is not None and any(holder):
                run.set(usage_metadata={"input_tokens": holder[0], "output_tokens": holder[1], "total_tokens": sum(holder)})
            return text
        finally:
            _CALL_TOKENS.reset(token)

    async def _call_with_retries(self, *, task: LLMTask, system: str, prompt: str, json_mode: bool) -> str:
        if not self.breaker.allow():
            raise LLMUnavailableError(f"{self.provider} circuit open ({self.breaker.reason or 'cooling down'})")
        delay = 1.5
        last: Exception | None = None
        for attempt in range(1, self.max_retries + 2):
            waited = time.monotonic()
            await self.limiter.acquire()
            started = time.monotonic()
            self.usage.limiter_wait_seconds += started - waited
            if attempt > 1:
                self.usage.retries += 1
            try:
                self.calls_made += 1
                self.usage.calls += 1
                text = await asyncio.wait_for(
                    self._complete(system=system, prompt=prompt, json_mode=json_mode), timeout=self.timeout_s
                )
                self.usage.provider_seconds += time.monotonic() - started
                self.breaker.record_success()
                return text
            except TimeoutError as exc:
                self.usage.provider_seconds += time.monotonic() - started
                last = LLMError(f"timeout after {self.timeout_s}s", retryable=True)
                logger.warning("llm %s timeout (attempt %d): %s", task, attempt, exc)
            except LLMError as exc:
                self.usage.provider_seconds += time.monotonic() - started
                if exc.is_rate_limit:
                    self.usage.rate_limited_attempts += 1
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

    def _record_surfaced(self, exc: LLMError) -> None:
        self.usage.surfaced_errors += 1
        if exc.is_infrastructure:
            self.usage.surfaced_infra_errors += 1
            self.usage.last_infra_error = str(exc)[:500]
        else:
            self.usage.surfaced_model_errors += 1

    async def generate_text(self, *, task: LLMTask, system: str, prompt: str) -> str:
        try:
            return (await self._call(task=task, system=system, prompt=prompt, json_mode=False)).strip()
        except LLMError as exc:
            self._record_surfaced(exc)
            raise

    async def generate_structured(
        self, *, task: LLMTask, system: str, prompt: str, schema: type[T], repair_attempts: int = 2
    ) -> T:
        """JSON-mode generation validated by Pydantic; validation errors are fed back for self-repair."""
        try:
            return await self._generate_structured(
                task=task, system=system, prompt=prompt, schema=schema, repair_attempts=repair_attempts
            )
        except LLMError as exc:
            self._record_surfaced(exc)
            raise

    async def _generate_structured(
        self, *, task: LLMTask, system: str, prompt: str, schema: type[T], repair_attempts: int
    ) -> T:
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
            "usage": self.usage.as_dict(),
        }


def tracing_is_enabled() -> bool:
    """LangSmith tracing active for this context (env-enabled or inside a traced parent run)."""
    from langsmith.utils import tracing_is_enabled as _enabled

    return bool(_enabled())


# Per-call token accumulator (a context variable, so concurrent calls on one client never mix counts).
_CALL_TOKENS: ContextVar[list[int] | None] = ContextVar("sentinel_llm_call_tokens", default=None)


def usage_from_mapping(data: Mapping[str, Any] | None, input_key: str, output_key: str) -> tuple[int | None, int | None]:
    """Read token counts from a provider usage mapping; None when absent (never guessed)."""
    if not data:
        return None, None
    i, o = data.get(input_key), data.get(output_key)
    return (int(i) if isinstance(i, int) else None), (int(o) if isinstance(o, int) else None)
