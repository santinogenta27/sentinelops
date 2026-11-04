"""LangSmith tracing is opt-in: on only with LANGSMITH_TRACING=true AND an API key."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest

from app.config import Settings
from app.llm.base import tracing_is_enabled
from app.observability import configure_tracing
from app.services.runtime import SentinelRuntime


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "LANGSMITH_API_KEY", "LANGSMITH_PROJECT", "LANGSMITH_ENDPOINT"):
        monkeypatch.setenv(name, "")  # records the original value, so whatever a test exports is undone
        monkeypatch.delenv(name)


def test_no_variables_means_no_tracing() -> None:
    assert configure_tracing(Settings(_env_file=None)) is False  # type: ignore[call-arg]
    assert os.environ["LANGSMITH_TRACING"] == "false"
    assert not tracing_is_enabled()


def test_tracing_without_key_stays_off(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="sentinelops.observability"):
        assert configure_tracing(Settings(_env_file=None, langsmith_tracing=True)) is False  # type: ignore[call-arg]
    assert "LANGSMITH_API_KEY is empty" in caplog.text
    assert not tracing_is_enabled()


def test_tracing_with_key_exports_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    s = Settings(_env_file=None, langsmith_tracing=True, langsmith_api_key="lsv2_test", langsmith_project="sentinel-test")  # type: ignore[call-arg]
    assert configure_tracing(s) is True
    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["LANGSMITH_API_KEY"] == "lsv2_test"
    assert os.environ["LANGSMITH_PROJECT"] == "sentinel-test"
    assert tracing_is_enabled()
    configure_tracing(Settings(_env_file=None))  # type: ignore[call-arg]  # leave the process clean


async def test_runtime_labels_runs_for_traces(tmp_path: Path) -> None:
    rt = await SentinelRuntime.create(Settings(_env_file=None, data_dir=tmp_path), llm=None, persist_vectors=False)  # type: ignore[call-arg]
    try:
        cfg = rt._config("TCK-1")
        assert cfg["configurable"] == {"thread_id": "TCK-1"}
        assert cfg["metadata"] == {"ticket_id": "TCK-1", "llm_provider": "deterministic", "llm_model": None}
        assert not tracing_is_enabled()
    finally:
        await rt.close()
