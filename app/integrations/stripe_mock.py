"""Embedded Mock Stripe engine.

Simulates refunds against the local ledger with the guarantees a real PSP gives:
idempotency keys, refund <= remaining captured amount, and transient failures.
No real money ever moves.
"""

from __future__ import annotations

import secrets

from app.db.database import Database, iso, utcnow
from app.integrations.faults import FaultInjector, ToolError
from app.models.schemas import RefundReceipt


class StripeMock:
    TOOL_NAME = "stripe.create_refund"

    def __init__(self, db: Database, faults: FaultInjector) -> None:
        self.db = db
        self.faults = faults

    async def create_refund(
        self,
        *,
        transaction_id: str,
        amount: float,
        currency: str,
        idempotency_key: str,
        ticket_id: str,
        approved_by: str | None,
    ) -> RefundReceipt:
        self.faults.check(self.TOOL_NAME)
        amount = round(amount, 2)
        if amount <= 0:
            raise ToolError("amount must be positive", retryable=False)

        async with self.db.connect() as conn:
            await conn.execute("BEGIN IMMEDIATE")
            try:
                cur = await conn.execute(
                    "SELECT id, transaction_id, amount, currency FROM refunds WHERE idempotency_key = ?",
                    (idempotency_key,),
                )
                existing = await cur.fetchone()
                if existing is not None:
                    await conn.rollback()
                    return RefundReceipt(
                        refund_id=existing["id"],
                        transaction_id=existing["transaction_id"],
                        amount=float(existing["amount"]),
                        currency=existing["currency"],
                        status="duplicate_ignored",
                        idempotency_key=idempotency_key,
                    )

                cur = await conn.execute(
                    "SELECT amount, refunded_amount, currency, status FROM transactions WHERE id = ?",
                    (transaction_id,),
                )
                txn = await cur.fetchone()
                if txn is None:
                    raise ToolError(f"no such charge: {transaction_id}", retryable=False)
                if txn["currency"] != currency:
                    raise ToolError("currency mismatch", retryable=False)
                remaining = round(float(txn["amount"]) - float(txn["refunded_amount"]), 2)
                if amount > remaining + 1e-9:
                    raise ToolError(
                        f"refund {amount:.2f} exceeds refundable balance {remaining:.2f} for {transaction_id}",
                        retryable=False,
                    )
                refund_id = f"re_{secrets.token_hex(8)}"
                new_refunded = round(float(txn["refunded_amount"]) + amount, 2)
                new_status = "REFUNDED" if new_refunded >= float(txn["amount"]) - 1e-9 else "PARTIALLY_REFUNDED"
                await conn.execute(
                    "INSERT INTO refunds (id, transaction_id, ticket_id, amount, currency, status, idempotency_key, approved_by, created_at) "
                    "VALUES (?, ?, ?, ?, ?, 'succeeded', ?, ?, ?)",
                    (refund_id, transaction_id, ticket_id, amount, currency, idempotency_key, approved_by, iso(utcnow())),
                )
                await conn.execute(
                    "UPDATE transactions SET refunded_amount = ?, status = ? WHERE id = ?",
                    (new_refunded, new_status, transaction_id),
                )
                await conn.commit()
            except BaseException:
                await conn.rollback()
                raise
        return RefundReceipt(
            refund_id=refund_id,
            transaction_id=transaction_id,
            amount=amount,
            currency=currency,
            status="succeeded",
            idempotency_key=idempotency_key,
        )

    async def freeze_card(self, *, card_last4: str, customer_id: str, ticket_id: str) -> None:
        self.faults.check("stripe.freeze_card")
        async with self.db.connect() as conn:
            await conn.execute(
                "INSERT OR IGNORE INTO card_freezes (card_last4, customer_id, ticket_id, created_at) VALUES (?, ?, ?, ?)",
                (card_last4, customer_id, ticket_id, iso(utcnow())),
            )
            await conn.commit()
