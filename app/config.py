"""Centralised, typed runtime configuration (12-factor, env-driven)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM ---
    llm_provider: Literal["gemini", "groq", "mock"] = "mock"
    gemini_api_key: SecretStr | None = None
    gemini_model: str = "gemini-2.5-flash"
    groq_api_key: SecretStr | None = None
    groq_model: str = "llama-3.3-70b-versatile"
    llm_rpm: int = Field(default=10, ge=1, le=1000)
    llm_timeout_s: float = Field(default=45.0, gt=0)
    llm_max_retries: int = Field(default=4, ge=0, le=10)

    # --- Embeddings ---
    embedding_provider: Literal["local", "gemini"] = "local"
    gemini_embedding_model: str = "gemini-embedding-001"

    # --- Business rules ---
    hitl_threshold_usd: float = Field(default=200.0, ge=0)
    judge_pass_threshold: float = Field(default=0.85, ge=0, le=1)
    max_tool_retries: int = Field(default=3, ge=1, le=10)
    max_redrafts: int = Field(default=2, ge=0, le=5)

    # --- Storage ---
    data_dir: Path = PROJECT_ROOT / "data"

    @property
    def business_db_path(self) -> Path:
        return self.data_dir / "sentinelops.db"

    @property
    def checkpoint_db_path(self) -> Path:
        return self.data_dir / "checkpoints.db"

    @property
    def chroma_path(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def policies_path(self) -> Path:
        return PROJECT_ROOT / "app" / "rag" / "policies.md"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return settings
