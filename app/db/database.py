"""Async SQLite business database (customers, ledger, refunds, CRM, outbox, tickets)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import aiosqlite

SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS customers (
    id TEXT PRIMARY KEY,
    email TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    tier TEXT NOT NULL CHECK (tier IN ('ENTERPRISE','PRO','FREE')),
    country TEXT NOT NULL,
    account_created_at TEXT NOT NULL,
    chargeback_count INTEGER NOT NULL DEFAULT 0,
    account_status TEXT NOT NULL DEFAULT 'ACTIVE'
);

CREATE TABLE IF NOT EXISTS transactions (
    id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL REFERENCES customers(id),
    amount REAL NOT NULL CHECK (amount >= 0),
    currency TEXT NOT NULL DEFAULT 'USD',
    status TEXT NOT NULL,
    gateway_status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    merchant_descriptor TEXT NOT NULL,
    card_last4 TEXT NOT NULL,
    ip_country TEXT NOT NULL,
    refunded_amount REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_txn_customer ON transactions(customer_id, created_at);

CREATE TABLE IF NOT EXISTS refunds (
    id TEXT PRIMARY KEY,
    transaction_id TEXT NOT NULL REFERENCES transactions(id),
    ticket_id TEXT NOT NULL,
    amount REAL NOT NULL CHECK (amount > 0),
    currency TEXT NOT NULL,
    status TEXT NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE,
    approved_by TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS card_freezes (
    card_last4 TEXT NOT NULL,
    customer_id TEXT NOT NULL,
    ticket_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (card_last4, customer_id)
);

CREATE TABLE IF NOT EXISTS crm_notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id TEXT,
    ticket_id TEXT NOT NULL,
    note TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS email_outbox (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id TEXT NOT NULL UNIQUE,
    to_address TEXT,
    subject TEXT NOT NULL,
    body TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tickets (
    id TEXT PRIMARY KEY,
    raw_message TEXT NOT NULL,
    customer_email TEXT,
    execution_status TEXT NOT NULL,
    summary_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ticket_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id TEXT NOT NULL,
    ts TEXT NOT NULL,
    type TEXT NOT NULL,
    node TEXT,
    payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_events_ticket ON ticket_events(ticket_id, id);

CREATE TABLE IF NOT EXISTS inbound_emails (
    message_id TEXT PRIMARY KEY,
    ticket_id TEXT NOT NULL,
    from_address TEXT NOT NULL,
    from_name TEXT NOT NULL DEFAULT '',
    subject TEXT NOT NULL DEFAULT '',
    body TEXT NOT NULL,
    source TEXT NOT NULL,
    received_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS supervisor_alerts (
    ticket_id TEXT PRIMARY KEY,
    recipient TEXT NOT NULL,
    subject TEXT NOT NULL,
    body_html TEXT NOT NULL,
    amount REAL NOT NULL,
    risk_level TEXT,
    channel TEXT NOT NULL,
    delivery TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""

# Additive, idempotent column migrations for databases created by earlier versions.
COLUMN_MIGRATIONS: dict[str, dict[str, str]] = {
    "tickets": {"channel": "TEXT NOT NULL DEFAULT 'web'", "subject": "TEXT"},
    "email_outbox": {
        "body_html": "TEXT",
        "delivery": "TEXT NOT NULL DEFAULT 'outbox'",
        "channel": "TEXT NOT NULL DEFAULT 'web'",
        "in_reply_to": "TEXT",
    },
}


def utcnow() -> datetime:
    return datetime.now(UTC)


def iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat()


class Database:
    """Thin async wrapper; one short-lived connection per unit of work (safe under concurrency)."""

    def __init__(self, path: Path) -> None:
        self.path = path

    @asynccontextmanager
    async def connect(self) -> AsyncIterator[aiosqlite.Connection]:
        conn = await aiosqlite.connect(self.path, timeout=30)
        conn.row_factory = aiosqlite.Row
        try:
            await conn.execute("PRAGMA foreign_keys=ON")
            yield conn
        finally:
            await conn.close()

    async def init_schema(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        async with self.connect() as conn:
            await conn.executescript(SCHEMA)
            for table, columns in COLUMN_MIGRATIONS.items():
                cur = await conn.execute(f"PRAGMA table_info({table})")
                existing = {row["name"] for row in await cur.fetchall()}
                for column, ddl in columns.items():
                    if column not in existing:
                        await conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
            await conn.commit()

    # --- tickets ------------------------------------------------------------
    async def upsert_ticket(
        self,
        ticket_id: str,
        raw_message: str,
        customer_email: str | None,
        execution_status: str,
        summary: dict[str, object] | None = None,
        *,
        channel: str | None = None,
        subject: str | None = None,
    ) -> None:
        now = iso(utcnow())
        async with self.connect() as conn:
            await conn.execute(
                """
                INSERT INTO tickets (id, raw_message, customer_email, execution_status, summary_json, created_at, updated_at,
                                     channel, subject)
                VALUES (?, ?, ?, ?, ?, ?, ?, COALESCE(?, 'web'), ?)
                ON CONFLICT(id) DO UPDATE SET
                    customer_email = COALESCE(excluded.customer_email, tickets.customer_email),
                    execution_status = excluded.execution_status,
                    summary_json = excluded.summary_json,
                    updated_at = excluded.updated_at,
                    channel = COALESCE(?, tickets.channel),
                    subject = COALESCE(excluded.subject, tickets.subject)
                """,
                (ticket_id, raw_message, customer_email, execution_status, json.dumps(summary or {}), now, now,
                 channel, subject, channel),
            )
            await conn.commit()

    async def list_tickets(self, limit: int = 100) -> list[dict[str, object]]:
        async with self.connect() as conn:
            cur = await conn.execute(
                "SELECT id, raw_message, customer_email, execution_status, summary_json, created_at, updated_at, "
                "channel, subject FROM tickets ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
            rows = await cur.fetchall()
        out: list[dict[str, object]] = []
        for r in rows:
            summary = json.loads(r["summary_json"])
            out.append(
                {
                    "ticket_id": r["id"],
                    "raw_message": r["raw_message"],
                    "customer_email": r["customer_email"],
                    "execution_status": r["execution_status"],
                    "created_at": r["created_at"],
                    "updated_at": r["updated_at"],
                    "channel": r["channel"],
                    "subject": r["subject"],
                    **summary,
                }
            )
        return out

    # --- inbound email ------------------------------------------------------
    async def record_inbound_email(
        self,
        *,
        message_id: str,
        ticket_id: str,
        from_address: str,
        from_name: str,
        subject: str,
        body: str,
        source: str,
    ) -> str | None:
        """Insert idempotently. Returns the *existing* ticket id if this Message-Id was already ingested."""
        async with self.connect() as conn:
            cur = await conn.execute(
                "INSERT OR IGNORE INTO inbound_emails (message_id, ticket_id, from_address, from_name, subject, body, source, received_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (message_id, ticket_id, from_address, from_name, subject, body, source, iso(utcnow())),
            )
            await conn.commit()
            if (cur.rowcount or 0) > 0:
                return None
            cur = await conn.execute("SELECT ticket_id FROM inbound_emails WHERE message_id = ?", (message_id,))
            row = await cur.fetchone()
        return str(row["ticket_id"]) if row else None

    async def list_inbound_emails(self, limit: int = 100) -> list[dict[str, object]]:
        async with self.connect() as conn:
            cur = await conn.execute("SELECT * FROM inbound_emails ORDER BY received_at DESC LIMIT ?", (limit,))
            rows = await cur.fetchall()
        return [{k: r[k] for k in r.keys()} for r in rows]

    async def list_supervisor_alerts(self, limit: int = 100) -> list[dict[str, object]]:
        async with self.connect() as conn:
            cur = await conn.execute("SELECT * FROM supervisor_alerts ORDER BY created_at DESC LIMIT ?", (limit,))
            rows = await cur.fetchall()
        return [{k: r[k] for k in r.keys()} for r in rows]

    async def get_ticket_row(self, ticket_id: str) -> dict[str, object] | None:
        async with self.connect() as conn:
            cur = await conn.execute(
                "SELECT id, raw_message, customer_email, execution_status, created_at, updated_at FROM tickets WHERE id = ?",
                (ticket_id,),
            )
            r = await cur.fetchone()
        if r is None:
            return None
        return {k: r[k] for k in r.keys()}

    # --- events -------------------------------------------------------------
    async def append_event(self, ticket_id: str, type_: str, node: str | None, payload: dict[str, object]) -> int:
        async with self.connect() as conn:
            cur = await conn.execute(
                "INSERT INTO ticket_events (ticket_id, ts, type, node, payload_json) VALUES (?, ?, ?, ?, ?)",
                (ticket_id, iso(utcnow()), type_, node, json.dumps(payload, default=str)),
            )
            await conn.commit()
            return int(cur.lastrowid or 0)

    async def list_events(self, ticket_id: str) -> list[dict[str, object]]:
        async with self.connect() as conn:
            cur = await conn.execute(
                "SELECT id, ts, type, node, payload_json FROM ticket_events WHERE ticket_id = ? ORDER BY id",
                (ticket_id,),
            )
            rows = await cur.fetchall()
        return [
            {"id": r["id"], "ts": r["ts"], "type": r["type"], "node": r["node"], "payload": json.loads(r["payload_json"])}
            for r in rows
        ]

    # --- ledger views ---------------------------------------------------------
    async def list_refunds(self) -> list[dict[str, object]]:
        async with self.connect() as conn:
            cur = await conn.execute("SELECT * FROM refunds ORDER BY created_at DESC")
            rows = await cur.fetchall()
        return [{k: r[k] for k in r.keys()} for r in rows]

    async def list_outbox(self) -> list[dict[str, object]]:
        async with self.connect() as conn:
            cur = await conn.execute("SELECT * FROM email_outbox ORDER BY id DESC")
            rows = await cur.fetchall()
        return [{k: r[k] for k in r.keys()} for r in rows]

    async def list_customers(self) -> list[dict[str, object]]:
        async with self.connect() as conn:
            cur = await conn.execute("SELECT id, email, name, tier, country, chargeback_count FROM customers ORDER BY id")
            rows = await cur.fetchall()
        return [{k: r[k] for k in r.keys()} for r in rows]
