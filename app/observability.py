"""Opt-in LangSmith tracing for the LangGraph agent.

LangChain/LangGraph and the ``langsmith`` SDK read their tracing switches from the process
environment, while :class:`~app.config.Settings` also reads ``.env`` (which pydantic-settings does
not export). This module reconciles both: tracing is turned on only when ``LANGSMITH_TRACING=true``
**and** ``LANGSMITH_API_KEY`` are configured; in every other case it is forced off, so a half-set
configuration never sends data or spams 401 warnings. With no LangSmith variables at all the
agent behaves exactly as before.
"""

from __future__ import annotations

import logging
import os

from app.config import Settings

logger = logging.getLogger("sentinelops.observability")

_SWITCHES = ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2")


def configure_tracing(settings: Settings) -> bool:
    """Apply the LangSmith configuration to the process environment. Returns True when tracing is on."""
    key = settings.langsmith_api_key.get_secret_value().strip() if settings.langsmith_api_key else ""
    if settings.langsmith_tracing and key:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_API_KEY"] = key
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
        if settings.langsmith_endpoint:
            os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
        _reset_langsmith_cache()
        logger.info("LangSmith tracing enabled (project=%s)", settings.langsmith_project)
        return True
    if settings.langsmith_tracing and not key:
        logger.warning("LANGSMITH_TRACING=true but LANGSMITH_API_KEY is empty; tracing stays disabled")
    for name in _SWITCHES:
        os.environ[name] = "false"
    _reset_langsmith_cache()
    return False


def _reset_langsmith_cache() -> None:
    """langsmith caches env lookups; clear them so a toggle at runtime takes effect."""
    try:
        from langsmith import utils

        for fn in (utils.get_env_var, utils.get_tracer_project):
            getattr(fn, "cache_clear")()  # noqa: B009 - lru_cache API, hidden by the SDK's overloads
    except (ImportError, AttributeError):  # pragma: no cover - older/newer SDK layouts
        pass
