"""Background IMAP worker (Gmail / Outlook / any IMAP4-SSL server).

Polls a folder (or Gmail label) for UNSEEN mail whose subject matches the dispute keywords,
turns each one into a ticket and only then marks it \\Seen — so a crash between fetch and
ingest never loses an email (re-ingest is idempotent through the Message-Id).
Non-matching mail is left untouched (read with BODY.PEEK, never marked as read).

One authenticated session is reused across polls (re-SELECT each poll so new mail is seen):
logging in every few seconds gets throttled by Gmail. The session is dropped on any error and
recycled every ``RECONNECT_AFTER_S``. The listener can be reconfigured and restarted at runtime
(mailbox connected from the cockpit) without restarting the server.
"""

from __future__ import annotations

import asyncio
import imaplib
import logging
import re
import threading
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any, Protocol

from app.config import Settings
from app.db.database import iso, utcnow
from app.integrations.email_inbound import InboundEmail, is_auto_generated, parse_rfc822

logger = logging.getLogger("sentinelops.imap")


class ImapClient(Protocol):
    def login(self, user: str, password: str) -> tuple[str, Any]: ...
    def select(self, mailbox: str = "INBOX", readonly: bool = False) -> tuple[str, Any]: ...
    def uid(self, command: str, *args: str) -> tuple[str, Any]: ...
    def logout(self) -> tuple[str, Any]: ...


IngestFn = Callable[[InboundEmail], Coroutine[Any, Any, object]]


@dataclass(slots=True)
class ListenerStatus:
    enabled: bool
    running: bool = False
    last_poll_at: str | None = None
    last_error: str | None = None
    polls: int = 0
    ingested: int = 0
    skipped: int = 0
    recent: list[str] = field(default_factory=list)


def _close_quietly(client: ImapClient) -> None:
    try:
        client.logout()
    except Exception:  # noqa: BLE001 - best effort
        pass


class EmailListener:
    RECONNECT_AFTER_S = 600.0
    MAX_BATCH = 25

    def __init__(
        self,
        settings: Settings,
        ingest: IngestFn,
        client_factory: Callable[[], ImapClient] | None = None,
    ) -> None:
        self.settings = settings
        self._ingest = ingest
        self._factory = client_factory or self._default_client
        self._task: asyncio.Task[None] | None = None
        self.status = ListenerStatus(enabled=settings.enable_email_listener)
        self._conn: ImapClient | None = None
        self._conn_since = 0.0
        self._conn_lock = threading.Lock()

    def _default_client(self) -> ImapClient:
        return imaplib.IMAP4_SSL(self.settings.imap_host, self.settings.imap_port, timeout=30)

    def _credentials(self) -> tuple[str, str]:
        user, secret = self.settings.imap_user, self.settings.imap_password
        if not user or secret is None or not secret.get_secret_value():
            raise RuntimeError("IMAP_USER / IMAP_PASSWORD are not configured")
        return user, secret.get_secret_value()

    def _matches(self, subject: str) -> bool:
        lowered = subject.lower()
        return any(k in lowered for k in self.settings.subject_keywords)

    @staticmethod
    def _raw_message(fetched: Any) -> bytes | None:
        for part in fetched or []:
            if isinstance(part, tuple) and len(part) >= 2 and isinstance(part[1], bytes):
                return part[1]
        return None

    # ------------------------------------------------------------- session
    def _drop_connection(self) -> None:
        conn, self._conn = self._conn, None
        if conn is not None:
            _close_quietly(conn)

    def _session(self) -> ImapClient:
        if self._conn is not None and time.monotonic() - self._conn_since > self.RECONNECT_AFTER_S:
            self._drop_connection()
        if self._conn is None:
            user, password = self._credentials()
            client = self._factory()
            try:
                client.login(user, password)
            except Exception:
                _close_quietly(client)
                raise
            self._conn, self._conn_since = client, time.monotonic()
        return self._conn

    def _poll_sync(self, loop: asyncio.AbstractEventLoop) -> int:
        """Blocking IMAP poll (runs in a worker thread). Returns the number of tickets created."""
        with self._conn_lock:
            try:
                return self._poll_session(self._session(), loop)
            except Exception:
                self._drop_connection()  # reconnect cleanly on the next poll
                raise

    def _poll_session(self, client: ImapClient, loop: asyncio.AbstractEventLoop) -> int:
        created = 0
        status, _ = client.select(self.settings.imap_folder)  # re-SELECT: the server reports new mail
        if status != "OK":
            raise RuntimeError(f"cannot select folder {self.settings.imap_folder!r}")
        status, data = client.uid("SEARCH", "UNSEEN")
        if status != "OK":
            raise RuntimeError("IMAP SEARCH failed")
        uids = [u for u in (data[0] or b"").decode().split() if re.fullmatch(r"\d+", u)]
        for uid in uids[: self.MAX_BATCH]:  # bounded batch per poll
            status, fetched = client.uid("FETCH", uid, "(BODY.PEEK[])")
            raw = self._raw_message(fetched) if status == "OK" else None
            if raw is None:
                continue
            try:
                msg, email = parse_rfc822(raw)
            except ValueError as exc:
                logger.warning("skipping unparseable message uid=%s: %s", uid, exc)
                self.status.skipped += 1
                continue
            if is_auto_generated(msg) or not self._matches(email.subject):
                self.status.skipped += 1
                continue  # left UNSEEN for the humans
            future = asyncio.run_coroutine_threadsafe(self._ingest(email), loop)
            try:
                future.result(timeout=60)
            except ValueError as exc:  # e.g. loop protection: leave it unread
                logger.info("not ingesting uid=%s: %s", uid, exc)
                self.status.skipped += 1
                continue
            client.uid("STORE", uid, "+FLAGS", "(\\Seen)")
            created += 1
            self.status.recent = ([f"{email.from_address}: {email.subject}"] + self.status.recent)[:10]
        return created

    # ---------------------------------------------------------------- loop
    async def poll_once(self) -> int:
        loop = asyncio.get_running_loop()
        try:
            created = await asyncio.to_thread(self._poll_sync, loop)
            self.status.ingested += created
            self.status.last_error = None
            return created
        except Exception as exc:  # noqa: BLE001 - the worker must survive provider errors
            self.status.last_error = f"{type(exc).__name__}: {str(exc)[:200]}"
            logger.warning("IMAP poll failed: %s", self.status.last_error)
            return 0
        finally:
            self.status.polls += 1
            self.status.last_poll_at = iso(utcnow())

    async def _run(self) -> None:
        self.status.running = True
        backoff = self.settings.email_poll_interval_s
        try:
            while True:
                await self.poll_once()
                failed = self.status.last_error is not None
                backoff = min(backoff * 2, 600) if failed else self.settings.email_poll_interval_s
                await asyncio.sleep(backoff)
        finally:
            self.status.running = False

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="imap-listener")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        await asyncio.to_thread(self._close_session)

    def _close_session(self) -> None:
        with self._conn_lock:
            self._drop_connection()

    async def restart(self) -> None:
        """Apply new settings (e.g. a mailbox connected from the cockpit) without restarting uvicorn."""
        await self.stop()
        self.status = ListenerStatus(enabled=self.settings.enable_email_listener)
        if self.status.enabled:
            self.start()

    def describe(self) -> dict[str, object]:
        user = self.settings.imap_user or ""
        masked = (user[:2] + "***" + user[user.find("@"):]) if "@" in user else ("***" if user else None)
        return {
            "enabled": self.status.enabled,
            "running": self.status.running,
            "host": self.settings.imap_host,
            "user": masked,
            "folder": self.settings.imap_folder,
            "keywords": self.settings.subject_keywords,
            "poll_interval_s": self.settings.email_poll_interval_s,
            "last_poll_at": self.status.last_poll_at,
            "last_error": self.status.last_error,
            "polls": self.status.polls,
            "ingested": self.status.ingested,
            "skipped": self.status.skipped,
            "recent": self.status.recent,
        }
