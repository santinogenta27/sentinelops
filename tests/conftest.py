"""Shared fixtures: every test gets an isolated data dir, ledger, checkpointer and fault injector."""

from __future__ import annotations

import os
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Hermetic tests: real process env vars override the developer's .env. The suite must never reach
# the real Stripe account, and Auto-Pilot is opted into explicitly by the tests that exercise it.
os.environ["STRIPE_API_KEY"] = ""
os.environ["AUTOPILOT_DEFAULT"] = "false"
os.environ["ENABLE_EMAIL_LISTENER"] = "false"
os.environ["ENABLE_EMAIL_SENDER"] = "false"

from app.config import Settings  # noqa: E402
from app.integrations.faults import FaultInjector  # noqa: E402
from app.llm.base import LLMClient, LLMError, LLMTask, T  # noqa: E402
from app.services.runtime import SentinelRuntime  # noqa: E402


def make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    return Settings(_env_file=None, llm_provider="mock", data_dir=tmp_path, **overrides)  # type: ignore[call-arg]


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return make_settings(tmp_path)


@pytest.fixture
def faults() -> FaultInjector:
    return FaultInjector()


@pytest.fixture
async def runtime(settings: Settings, faults: FaultInjector) -> AsyncIterator[SentinelRuntime]:
    rt = await SentinelRuntime.create(settings, llm=None, faults=faults, persist_vectors=False)
    try:
        yield rt
    finally:
        await rt.close()


class ScriptedLLM(LLMClient):
    """Test double: returns scripted objects per task (or raises), recording every call."""

    provider = "scripted"

    def __init__(self, script: dict[LLMTask, list[Any]]) -> None:
        super().__init__(model="scripted-1", rpm=1000, max_retries=0, timeout_s=5)
        self.script = {k: list(v) for k, v in script.items()}
        self.calls: list[tuple[LLMTask, str]] = []

    async def _complete(self, *, system: str, prompt: str, json_mode: bool) -> str:  # pragma: no cover
        raise NotImplementedError

    def _next(self, task: LLMTask, prompt: str) -> Any:
        self.calls.append((task, prompt))
        queue = self.script.get(task)
        if not queue:
            raise LLMError(f"no scripted response for {task}", retryable=False)
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(item, Exception):
            raise item
        return item

    async def generate_structured(self, *, task: LLMTask, system: str, prompt: str, schema: type[T], repair_attempts: int = 2) -> T:
        return schema.model_validate(self._next(task, prompt))

    async def generate_text(self, *, task: LLMTask, system: str, prompt: str) -> str:
        return str(self._next(task, prompt))
