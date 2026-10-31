"""Typed read-side repository. Every row leaves this module as a validated Pydantic model."""

from __future__ import annotations

from datetime import datetime, timedelta

import aiosqlite

from app.db.database import Database, iso, utcnow
from app.models.schemas import CustomerRecord, TransactionQuery, TransactionRecord


def _customer(row: aiosqlite.Row) -> CustomerRecord:
    return CustomerRecord(
        id=row["id"],
        email=row["email"],
        name=row["name"],
        tier=row["tier"],
        country=row["country"],
        account_created_at=datetime.fromisoformat(row["account_created_at"]),
        chargeback_count=int(row["chargeback_count"]),
        account_status=row["account_status"],
    )


def _transaction(row: aiosqlite.Row) -> TransactionRecord:
    return TransactionRecord(
        id=row["id"],
        customer_id=row["customer_id"],
        amount=float(row["amount"]),
        currency=row["currency"],
        status=row["status"],
        gateway_status=row["gateway_status"],
        created_at=datetime.fromisoformat(row["created_at"]),
        merchant_descriptor=row["merchant_descriptor"],
        card_last4=row["card_last4"],
        ip_country=row["ip_country"],
        refunded_amount=float(row["refunded_amount"]),
    )


class Repository:
    def __init__(self, db: Database) -> None:
        self.db = db

    async def get_customer_by_email(self, email: str) -> CustomerRecord | None:
        async with self.db.connect() as conn:
            cur = await conn.execute("SELECT * FROM customers WHERE email = ?", (email.lower(),))
            row = await cur.fetchone()
        return _customer(row) if row else None

    async def get_customer_by_id(self, customer_id: str) -> CustomerRecord | None:
        async with self.db.connect() as conn:
            cur = await conn.execute("SELECT * FROM customers WHERE id = ?", (customer_id,))
            row = await cur.fetchone()
        return _customer(row) if row else None

    async def get_transaction(self, transaction_id: str) -> TransactionRecord | None:
        async with self.db.connect() as conn:
            cur = await conn.execute("SELECT * FROM transactions WHERE id = ?", (transaction_id,))
            row = await cur.fetchone()
        return _transaction(row) if row else None

    async def query_transactions(self, query: TransactionQuery) -> tuple[CustomerRecord | None, list[TransactionRecord]]:
        """Resolve the customer and their transactions.

        Security: when both an email and transaction ids are given, transaction ids are
        scoped to that customer so a requester can never pull another customer's ledger.
        """
        since = iso(utcnow() - timedelta(days=query.lookback_days))
        customer: CustomerRecord | None = None
        if query.customer_email:
            customer = await self.get_customer_by_email(query.customer_email)

        elif query.transaction_ids:
            # Identify the owner through the referenced transactions (no email supplied).
            ids = list(query.transaction_ids)
            async with self.db.connect() as conn:
                cur = await conn.execute(
                    f"SELECT DISTINCT customer_id FROM transactions WHERE id IN ({','.join('?' * len(ids))})",  # noqa: S608
                    ids,
                )
                owners = [r["customer_id"] for r in await cur.fetchall()]
            if len(owners) == 1:  # ambiguous ownership -> refuse to mix ledgers
                customer = await self.get_customer_by_id(owners[0])

        if customer is None:
            return None, []
        async with self.db.connect() as conn:
            cur = await conn.execute(
                "SELECT * FROM transactions WHERE customer_id = ? AND created_at >= ? ORDER BY created_at",
                (customer.id, since),
            )
            rows = list(await cur.fetchall())
        return customer, [_transaction(r) for r in rows]
