"""Mock CRM webhook + transactional email outbox."""

from __future__ import annotations

from app.db.database import Database, iso, utcnow
from app.integrations.faults import FaultInjector


class CRMMock:
    def __init__(self, db: Database, faults: FaultInjector) -> None:
        self.db = db
        self.faults = faults

    async def add_note(self, *, customer_id: str | None, ticket_id: str, note: str) -> None:
        self.faults.check("crm.add_note")
        async with self.db.connect() as conn:
            await conn.execute(
                "INSERT INTO crm_notes (customer_id, ticket_id, note, created_at) VALUES (?, ?, ?, ?)",
                (customer_id, ticket_id, note[:2000], iso(utcnow())),
            )
            await conn.commit()

    async def send_email(self, *, ticket_id: str, to_address: str | None, subject: str, body: str) -> bool:
        """Idempotent per ticket: a resumed/replayed graph never emails the customer twice."""
        self.faults.check("email.send")
        async with self.db.connect() as conn:
            cur = await conn.execute(
                "INSERT OR IGNORE INTO email_outbox (ticket_id, to_address, subject, body, created_at) VALUES (?, ?, ?, ?, ?)",
                (ticket_id, to_address, subject, body, iso(utcnow())),
            )
            await conn.commit()
            return (cur.rowcount or 0) > 0
