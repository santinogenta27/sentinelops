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

    # --- Email channel: inbound webhook -------------------------------------
    support_email_address: str = "support@sentinelops.demo"
    supervisor_email: str = "billing-supervisor@sentinelops.demo"
    inbound_webhook_secret: SecretStr | None = None  # if set, X-Webhook-Token header is required

    # --- Email channel: IMAP listener (Gmail / Outlook) ----------------------
    enable_email_listener: bool = False
    imap_host: str = "imap.gmail.com"
    imap_port: int = Field(default=993, ge=1, le=65535)
    imap_user: str | None = None
    imap_password: SecretStr | None = None  # Gmail: use an App Password, never the account password
    imap_folder: str = "INBOX"  # a Gmail label is exposed as an IMAP folder, e.g. "Disputes"
    # Optional subject filter (comma-separated). Empty = every human email in the folder is a ticket;
    # newsletters, bounces, auto-replies and our own replies are always skipped.
    email_subject_keywords: str = ""
    email_poll_interval_s: int = Field(default=30, ge=5, le=3600)
    # Read mail (seen or unseen) that arrived in the last N minutes, so a message already opened on
    # the phone is still processed. Each message is ingested once (UID memory + Message-Id dedupe).
    email_lookback_minutes: int = Field(default=120, ge=1, le=10_080)

    # --- Email channel: SMTP replies (optional, real delivery) ---------------
    enable_email_sender: bool = False
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = Field(default=587, ge=1, le=65535)
    smtp_user: str | None = None  # defaults to imap_user
    smtp_password: SecretStr | None = None  # defaults to imap_password

    # --- Auto-Pilot (zero-touch) -------------------------------------------
    # When ON, the HumanGate approves payouts that comply with policy (no injection, sender owns the
    # account, risk not HIGH) up to ``autopilot_max_usd``. Everything else still waits for a human.
    autopilot_default: bool = True  # initial value; the cockpit toggle persists the live value in the DB
    autopilot_max_usd: float = Field(default=1000.0, ge=0, le=100_000)

    # --- Stripe (live dashboard + test-mode mirror of ledger refunds) -------
    stripe_api_key: SecretStr | None = None  # sk_test_... (writes are refused for live keys)
    stripe_mirror_refunds: bool = True  # mirror each ledger refund into Stripe TEST mode

    # --- Storage ---
    data_dir: Path = PROJECT_ROOT / "data"

    @property
    def subject_keywords(self) -> list[str]:
        return [k.strip().lower() for k in self.email_subject_keywords.split(",") if k.strip()]

    @property
    def smtp_credentials(self) -> tuple[str, str] | None:
        user = self.smtp_user or self.imap_user
        secret = self.smtp_password or self.imap_password
        if not user or secret is None or not secret.get_secret_value():
            return None
        return user, secret.get_secret_value()

    @property
    def stripe_key(self) -> str | None:
        key = self.stripe_api_key.get_secret_value().strip() if self.stripe_api_key else ""
        return key or None

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
