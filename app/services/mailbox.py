"""Connect a company support mailbox from the cockpit (no .env editing, no server restart).

The configuration is validated against the real IMAP/SMTP servers, persisted in the local
business database (``app_settings``; ``data/`` is git-ignored) and applied *in place* to the
shared ``Settings`` object, so the IMAP listener (restarted here) and the ``Mailer`` pick it up
immediately. The App Password is never returned by any endpoint.
"""

from __future__ import annotations

import asyncio
import imaplib
import logging
import re
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, StringConstraints, model_validator
from typing_extensions import Annotated

from app.config import Settings
from app.db.database import Database
from app.integrations.imap_listener import EmailListener, ImapClient
from app.integrations.mailer import smtp_session

logger = logging.getLogger("sentinelops.mailbox")

Provider = Literal["gmail", "outlook", "custom"]
MAILBOX_KEY = "mailbox"
EMAIL_RE = r"^[^@\s<>\"']{1,64}@[A-Za-z0-9.-]{1,190}\.[A-Za-z]{2,24}$"
HOST_RE = r"^[A-Za-z0-9.-]{3,253}$"

PROVIDERS: dict[str, dict[str, Any]] = {
    "gmail": {
        "label": "Gmail / Google Workspace",
        "imap_host": "imap.gmail.com", "imap_port": 993, "smtp_host": "smtp.gmail.com", "smtp_port": 587,
        "help_url": "https://myaccount.google.com/apppasswords",
        "help": "Activá la verificación en 2 pasos y creá una App Password (16 letras) en myaccount.google.com/apppasswords.",
    },
    "outlook": {
        "label": "Outlook / Office 365",
        "imap_host": "outlook.office365.com", "imap_port": 993, "smtp_host": "smtp.office365.com", "smtp_port": 587,
        "help_url": "https://account.live.com/proofs/AppPassword",
        "help": "Requiere IMAP/SMTP AUTH habilitado y una App Password. Muchos tenants de Microsoft 365 sólo aceptan OAuth2.",
    },
    "custom": {
        "label": "IMAP personalizado",
        "imap_host": None, "imap_port": 993, "smtp_host": None, "smtp_port": 587,
        "help_url": None,
        "help": "Servidor IMAP con SSL (puerto 993) y SMTP con STARTTLS (587) o SSL (465).",
    },
}

EmailStrT = Annotated[str, StringConstraints(strip_whitespace=True, to_lower=True, pattern=EMAIL_RE)]
HostT = Annotated[str, StringConstraints(strip_whitespace=True, pattern=HOST_RE)]


class MailboxConfigIn(BaseModel):
    """Payload of ``POST /api/email/test-connection`` and ``/api/email/save-config``."""

    model_config = ConfigDict(extra="forbid")

    provider: Provider = "gmail"
    email: EmailStrT
    app_password: Annotated[str, StringConstraints(min_length=4, max_length=256)] | None = None  # None = keep stored
    imap_host: HostT | None = None
    imap_port: int = Field(default=993, ge=1, le=65535)
    smtp_host: HostT | None = None
    smtp_port: int = Field(default=587, ge=1, le=65535)
    folder: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=120)] = "INBOX"
    poll_interval_s: int = Field(default=5, ge=5, le=3600)
    enable_sender: bool = True
    keywords: Annotated[str, StringConstraints(max_length=500)] | None = None
    # Demo helper: let a tester's real address act as a verified alias of a ledger customer.
    demo_alias_email: EmailStrT | None = None
    demo_alias_customer_id: Annotated[str, StringConstraints(max_length=64)] | None = None

    @model_validator(mode="after")
    def _resolve_hosts(self) -> MailboxConfigIn:
        preset = PROVIDERS[self.provider]
        if self.provider != "custom":
            self.imap_host, self.imap_port = preset["imap_host"], preset["imap_port"]
            self.smtp_host, self.smtp_port = preset["smtp_host"], preset["smtp_port"]
        elif not self.imap_host:
            raise ValueError("imap_host is required for a custom IMAP provider")
        if self.enable_sender and not self.smtp_host:
            raise ValueError("smtp_host is required to send replies (or disable enable_sender)")
        if self.app_password is not None:
            pw = self.app_password.strip()
            # Google shows App Passwords as "abcd efgh ijkl mnop"; the spaces are not part of it.
            self.app_password = re.sub(r"\s+", "", pw) if self.provider == "gmail" else pw
        if bool(self.demo_alias_email) != bool(self.demo_alias_customer_id):
            raise ValueError("demo_alias_email and demo_alias_customer_id go together")
        return self


ImapFactory = Callable[[str, int], ImapClient]
SmtpCheck = Callable[[str, int, str, str], None]


def _default_imap(host: str, port: int) -> ImapClient:
    return imaplib.IMAP4_SSL(host, port, timeout=15)


def _default_smtp_check(host: str, port: int, user: str, password: str) -> None:
    with smtp_session(host, port, timeout=15) as smtp:
        smtp.login(user, password)


def _friendly(exc: Exception, kind: str) -> str:
    text = str(exc)
    if isinstance(exc, bytes):  # pragma: no cover - defensive
        text = exc.decode(errors="replace")
    low = text.lower()
    if "authenticationfailed" in low or "invalid credentials" in low or "535" in low or "auth" in low and "fail" in low:
        return (f"{kind}: credenciales rechazadas. Usá una App Password (no tu contraseña normal) y verificá que "
                f"{'IMAP' if kind == 'IMAP' else 'SMTP AUTH'} esté habilitado en la cuenta.")
    if isinstance(exc, (TimeoutError, OSError)) and not isinstance(exc, imaplib.IMAP4.error):
        return f"{kind}: no se pudo conectar al servidor ({type(exc).__name__}: {text[:160]})"
    return f"{kind}: {type(exc).__name__}: {text[:200]}"


class MailboxService:
    def __init__(
        self,
        db: Database,
        settings: Settings,
        listener: EmailListener,
        *,
        imap_factory: ImapFactory | None = None,
        smtp_check: SmtpCheck | None = None,
    ) -> None:
        self.db = db
        self.settings = settings
        self.listener = listener
        self._imap = imap_factory or _default_imap
        self._smtp = smtp_check or _default_smtp_check
        self._saved: dict[str, Any] | None = None

    # -------------------------------------------------------------- persistence
    async def load_and_apply(self) -> bool:
        """Startup: re-apply a mailbox connected from the cockpit in a previous run."""
        saved = await self.db.get_setting(MAILBOX_KEY)
        if not isinstance(saved, dict) or not saved.get("connected"):
            return False
        self._saved = saved
        self._apply(saved)
        return True

    def _password_for(self, cfg: MailboxConfigIn) -> str:
        if cfg.app_password:
            return cfg.app_password
        saved = self._saved or {}
        if saved.get("email") == cfg.email and saved.get("app_password"):
            return str(saved["app_password"])
        raise ValueError("app_password is required")

    def _apply(self, saved: dict[str, Any]) -> None:
        s = self.settings
        s.imap_host = str(saved["imap_host"])
        s.imap_port = int(saved["imap_port"])
        s.imap_user = str(saved["email"])
        s.imap_password = SecretStr(str(saved["app_password"]))
        s.imap_folder = str(saved.get("folder") or "INBOX")
        s.email_poll_interval_s = int(saved.get("poll_interval_s") or 5)
        s.enable_email_listener = bool(saved.get("connected"))
        s.enable_email_sender = bool(saved.get("enable_sender")) and bool(saved.get("smtp_host"))
        if saved.get("smtp_host"):
            s.smtp_host = str(saved["smtp_host"])
            s.smtp_port = int(saved["smtp_port"])
        s.smtp_user = None  # fall back to the mailbox credentials
        s.smtp_password = None
        s.support_email_address = str(saved["email"])  # loop protection + reply address
        if saved.get("keywords"):
            s.email_subject_keywords = str(saved["keywords"])

    # --------------------------------------------------------------- operations
    def _check_sync(self, cfg: MailboxConfigIn, password: str) -> dict[str, Any]:
        out: dict[str, Any] = {"ok": False, "imap": {"ok": False}, "smtp": {"ok": None}}
        assert cfg.imap_host is not None
        client: ImapClient | None = None
        try:
            client = self._imap(cfg.imap_host, cfg.imap_port)
            client.login(cfg.email, password)
            status, data = client.select(cfg.folder, readonly=True)
            if status != "OK":
                raise RuntimeError(f"la carpeta/etiqueta {cfg.folder!r} no existe")
            first = data[0] if data else None
            total = int(first) if isinstance(first, bytes) and first.isdigit() else None
            status, found = client.uid("SEARCH", "UNSEEN")
            unseen = len((found[0] or b"").split()) if status == "OK" and found else 0
            out["imap"] = {"ok": True, "host": cfg.imap_host, "folder": cfg.folder, "messages": total, "unseen": unseen}
        except Exception as exc:  # noqa: BLE001 - reported to the operator
            out["imap"] = {"ok": False, "host": cfg.imap_host, "error": _friendly(exc, "IMAP")}
            return out
        finally:
            if client is not None:
                try:
                    client.logout()
                except Exception:  # noqa: BLE001
                    pass
        if cfg.enable_sender and cfg.smtp_host:
            try:
                self._smtp(cfg.smtp_host, cfg.smtp_port, cfg.email, password)
                out["smtp"] = {"ok": True, "host": cfg.smtp_host, "port": cfg.smtp_port}
            except Exception as exc:  # noqa: BLE001
                out["smtp"] = {"ok": False, "host": cfg.smtp_host, "error": _friendly(exc, "SMTP")}
        out["ok"] = bool(out["imap"]["ok"]) and out["smtp"]["ok"] is not False
        return out

    async def test(self, cfg: MailboxConfigIn) -> dict[str, Any]:
        password = self._password_for(cfg)
        return await asyncio.wait_for(asyncio.to_thread(self._check_sync, cfg, password), timeout=45)

    async def save(self, cfg: MailboxConfigIn) -> dict[str, Any]:
        """Validate against the real servers, persist, apply live and (re)start the listener."""
        password = self._password_for(cfg)
        check = await asyncio.wait_for(asyncio.to_thread(self._check_sync, cfg, password), timeout=45)
        if not check["ok"]:
            return {"saved": False, "check": check, "status": self.describe()}
        if cfg.demo_alias_email and cfg.demo_alias_customer_id:
            # The alias may be the support mailbox itself (demo self-send from the phone): our own
            # replies carry loop-protection markers, so answering ourselves never loops.
            await self.db.set_customer_alias(cfg.demo_alias_email, cfg.demo_alias_customer_id)
        saved = {
            "connected": True,
            "provider": cfg.provider,
            "email": cfg.email,
            "app_password": password,
            "imap_host": cfg.imap_host,
            "imap_port": cfg.imap_port,
            "smtp_host": cfg.smtp_host,
            "smtp_port": cfg.smtp_port,
            "folder": cfg.folder,
            "poll_interval_s": cfg.poll_interval_s,
            "enable_sender": cfg.enable_sender,
            "keywords": cfg.keywords,
        }
        await self.db.set_setting(MAILBOX_KEY, saved)
        self._saved = saved
        self._apply(saved)
        await self.listener.restart()
        logger.info("mailbox connected: %s via %s (poll %ss)", cfg.email, cfg.imap_host, cfg.poll_interval_s)
        return {"saved": True, "check": check, "status": self.describe()}

    async def disconnect(self) -> dict[str, Any]:
        if self._saved:
            self._saved = {**self._saved, "connected": False}
            await self.db.set_setting(MAILBOX_KEY, self._saved)
        self.settings.enable_email_listener = False
        self.settings.enable_email_sender = False
        await self.listener.restart()
        return self.describe()

    def describe(self) -> dict[str, Any]:
        saved = self._saved or {}
        listener = self.listener.describe()
        connected = bool(self.settings.enable_email_listener and self.settings.imap_user)
        return {
            "connected": connected,
            "provider": saved.get("provider") if saved.get("connected") else None,
            "email": self.settings.imap_user if connected else None,
            "has_saved_password": bool(saved.get("app_password")),
            "saved_email": saved.get("email"),
            "imap_host": self.settings.imap_host,
            "folder": self.settings.imap_folder,
            "poll_interval_s": self.settings.email_poll_interval_s,
            "sender_enabled": self.settings.enable_email_sender,
            "listening": bool(listener["running"]),
            "healthy": connected and listener["last_error"] is None,
            "listener": listener,
            "providers": PROVIDERS,
        }
