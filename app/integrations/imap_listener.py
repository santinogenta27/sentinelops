"""Background IMAP worker (Gmail / Outlook / any IMAP4-SSL server).

Every poll looks at the mail that arrived in the last ``EMAIL_LOOKBACK_MINUTES`` (seen *or* unseen:
a message already opened on the phone must still become a ticket), newest first. Each UID is
evaluated exactly once per session (in-memory memory) and ingestion is idempotent through the
Message-Id, so a restart never duplicates tickets. Ingested mail is then marked \\Seen.

Skipped, never ingested: bounces / auto-replies, newsletters and mailing lists, no-reply senders and
the replies SentinelOps itself sent (they land back in the inbox when the customer *is* the support
mailbox: demo self-send). An optional subject keyword filter can still be configured.

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
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol

from app.config import Settings
from app.db.database import iso, utcnow
from app.integrations.email_inbound import InboundEmail, auto_generated_reason, parse_rfc822

logger = logging.getLogger("sentinelops.imap")

_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
_INTERNALDATE_RE = re.compile(rb'UID (\d+).*?INTERNALDATE "([^"]+)"|INTERNALDATE "([^"]+)".*?UID (\d+)')


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


@dataclass(slots=True)
class PolledMessage:
    uid: int
    from_address: str
    from_name: str
    subject: str
    ticket_id: str | None = None
    duplicate: bool = False
    skipped_reason: str | None = None


@dataclass(slots=True)
class PollReport:
    new: list[PolledMessage] = field(default_factory=list)  # became a ticket now
    duplicates: list[PolledMessage] = field(default_factory=list)  # already a ticket (same Message-Id)
    skipped: list[PolledMessage] = field(default_factory=list)  # filtered (auto-generated, keyword…)
    scanned: int = 0  # UIDs in the lookback window looked at for the first time
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "found": len(self.new),
            "new": [asdict(m) for m in self.new],
            "duplicates": [asdict(m) for m in self.duplicates],
            "skipped": [asdict(m) for m in self.skipped],
            "scanned": self.scanned,
            "error": self.error,
        }


def _close_quietly(client: ImapClient) -> None:
    try:
        client.logout()
    except Exception:  # noqa: BLE001 - best effort
        pass


def imap_date(value: datetime) -> str:
    """IMAP SEARCH date (RFC 3501 ``date``), locale-independent."""
    return f"{value.day:02d}-{_MONTHS[value.month - 1]}-{value.year}"


def parse_internaldate(raw: str) -> datetime | None:
    """``"31-Oct-2026 14:20:01 -0300"`` → aware datetime (None if unparseable)."""
    m = re.fullmatch(r"\s*(\d{1,2})-(\w{3})-(\d{4}) (\d{2}):(\d{2}):(\d{2}) ([+-])(\d{2})(\d{2})\s*", raw)
    if not m or m.group(2).title() not in _MONTHS:
        return None
    day, mon, year, hh, mm, ss, sign, oh, om = m.groups()
    offset = timedelta(hours=int(oh), minutes=int(om)) * (1 if sign == "+" else -1)
    return datetime(int(year), _MONTHS.index(mon.title()) + 1, int(day), int(hh), int(mm), int(ss),
                    tzinfo=timezone(offset))


def _internaldates(fetched: Any) -> dict[int, datetime]:
    out: dict[int, datetime] = {}
    for part in fetched or []:
        line = part[0] if isinstance(part, tuple) else part
        if not isinstance(line, bytes):
            continue
        m = _INTERNALDATE_RE.search(line)
        if not m:
            continue
        uid = int(m.group(1) or m.group(4))
        when = parse_internaldate((m.group(2) or m.group(3)).decode(errors="replace"))
        if when is not None:
            out[uid] = when
    return out


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
        self.last_report: PollReport | None = None
        self._conn: ImapClient | None = None
        self._conn_since = 0.0
        self._conn_lock = threading.Lock()
        self._evaluated: set[int] = set()  # UIDs already decided (ingested / duplicate / skipped / too old)
        self._evaluated_key: tuple[str | None, str] | None = None

    def _default_client(self) -> ImapClient:
        return imaplib.IMAP4_SSL(self.settings.imap_host, self.settings.imap_port, timeout=30)

    def _credentials(self) -> tuple[str, str]:
        user, secret = self.settings.imap_user, self.settings.imap_password
        if not user or secret is None or not secret.get_secret_value():
            raise RuntimeError("IMAP_USER / IMAP_PASSWORD are not configured")
        return user, secret.get_secret_value()

    def _matches(self, subject: str) -> bool:
        """Optional subject filter: with no keywords configured every human email is accepted."""
        keywords = self.settings.subject_keywords
        lowered = subject.lower()
        return not keywords or any(k in lowered for k in keywords)

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

    def _poll_sync(self, loop: asyncio.AbstractEventLoop) -> PollReport:
        """Blocking IMAP poll (runs in a worker thread)."""
        with self._conn_lock:
            try:
                return self._poll_session(self._session(), loop)
            except Exception:
                self._drop_connection()  # reconnect cleanly on the next poll
                raise

    def _window_uids(self, client: ImapClient, cutoff: datetime) -> list[int]:
        """UIDs (seen or unseen) that arrived after ``cutoff`` and were not evaluated yet, newest first."""
        key = (self.settings.imap_user, self.settings.imap_folder)
        if key != self._evaluated_key:
            self._evaluated, self._evaluated_key = set(), key
        # SINCE has day granularity (and ignores time zones): ask one day earlier, refine with INTERNALDATE.
        status, data = client.uid("SEARCH", "SINCE", imap_date(cutoff - timedelta(days=1)))
        if status != "OK":
            raise RuntimeError("IMAP SEARCH failed")
        fresh = sorted(
            {int(u) for u in (data[0] or b"").decode().split() if re.fullmatch(r"\d+", u)} - self._evaluated,
            reverse=True,
        )
        if not fresh:
            return []
        status, fetched = client.uid("FETCH", ",".join(map(str, fresh)), "(UID INTERNALDATE)")
        dates = _internaldates(fetched) if status == "OK" else {}
        in_window: list[int] = []
        for uid in fresh:
            when = dates.get(uid)
            if when is not None and when < cutoff:
                self._evaluated.add(uid)  # older than the window: it only gets older
            else:
                in_window.append(uid)
        return in_window

    def _poll_session(self, client: ImapClient, loop: asyncio.AbstractEventLoop) -> PollReport:
        report = PollReport()
        status, _ = client.select(self.settings.imap_folder)  # re-SELECT: the server reports new mail
        if status != "OK":
            raise RuntimeError(f"cannot select folder {self.settings.imap_folder!r}")
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=self.settings.email_lookback_minutes)
        uids = self._window_uids(client, cutoff)
        report.scanned = len(uids)
        for uid in uids[: self.MAX_BATCH]:  # bounded batch per poll; the rest comes next poll
            status, fetched = client.uid("FETCH", str(uid), "(BODY.PEEK[])")
            raw = self._raw_message(fetched) if status == "OK" else None
            if raw is None:
                continue
            try:
                msg, email = parse_rfc822(raw)
            except ValueError as exc:
                logger.warning("skipping unparseable message uid=%s: %s", uid, exc)
                self._evaluated.add(uid)
                self.status.skipped += 1
                continue
            item = PolledMessage(uid, email.from_address, email.from_name, email.subject)
            reason = auto_generated_reason(msg)
            if reason is None and not self._matches(email.subject):
                reason = "el asunto no coincide con las palabras clave configuradas"
            if reason is not None:
                item.skipped_reason = reason
                report.skipped.append(item)
                self._evaluated.add(uid)
                self.status.skipped += 1
                continue  # left untouched (read with BODY.PEEK, flags unchanged)
            future = asyncio.run_coroutine_threadsafe(self._ingest(email), loop)
            try:
                result = future.result(timeout=60)
            except ValueError as exc:  # e.g. loop protection
                logger.info("not ingesting uid=%s: %s", uid, exc)
                item.skipped_reason = str(exc)
                report.skipped.append(item)
                self._evaluated.add(uid)
                self.status.skipped += 1
                continue
            item.ticket_id = getattr(result, "ticket_id", None)
            item.duplicate = bool(getattr(result, "duplicate", False))
            client.uid("STORE", str(uid), "+FLAGS", "(\\Seen)")
            self._evaluated.add(uid)
            if item.duplicate:
                report.duplicates.append(item)
                continue
            report.new.append(item)
            self.status.recent = ([f"{email.from_address}: {email.subject}"] + self.status.recent)[:10]
        return report

    # ---------------------------------------------------------------- loop
    async def poll(self) -> PollReport:
        """One poll; never raises (errors are reported and kept in ``status.last_error``)."""
        loop = asyncio.get_running_loop()
        try:
            report = await asyncio.to_thread(self._poll_sync, loop)
            self.status.ingested += len(report.new)
            self.status.last_error = None
        except Exception as exc:  # noqa: BLE001 - the worker must survive provider errors
            self.status.last_error = f"{type(exc).__name__}: {str(exc)[:200]}"
            logger.warning("IMAP poll failed: %s", self.status.last_error)
            report = PollReport(error=self.status.last_error)
        finally:
            self.status.polls += 1
            self.status.last_poll_at = iso(utcnow())
        self.last_report = report
        return report

    async def poll_once(self) -> int:
        """One poll; returns the number of tickets created."""
        return len((await self.poll()).new)

    async def _run(self) -> None:
        self.status.running = True
        backoff = self.settings.email_poll_interval_s
        try:
            while True:
                await self.poll()
                failed = self.status.last_error is not None
                backoff = min(backoff * 2, 600) if failed else self.settings.email_poll_interval_s
                await asyncio.sleep(backoff)
        finally:
            self.status.running = False

    def start(self) -> None:
        # The config may have been applied after construction (mailbox restored from the DB at startup).
        self.status.enabled = self.settings.enable_email_listener
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
            self._evaluated, self._evaluated_key = set(), None

    async def restart(self) -> None:
        """Apply new settings (e.g. a mailbox connected from the cockpit) without restarting uvicorn."""
        await self.stop()
        self.status = ListenerStatus(enabled=self.settings.enable_email_listener)
        self.last_report = None
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
            "lookback_minutes": self.settings.email_lookback_minutes,
            "poll_interval_s": self.settings.email_poll_interval_s,
            "last_poll_at": self.status.last_poll_at,
            "last_error": self.status.last_error,
            "polls": self.status.polls,
            "ingested": self.status.ingested,
            "skipped": self.status.skipped,
            "recent": self.status.recent,
        }
