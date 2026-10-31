"""Deterministic seed data. Timestamps are relative to seeding time so policy windows stay valid."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from app.db.database import Database, iso, utcnow


@dataclass(frozen=True, slots=True)
class SeedCustomer:
    id: str
    email: str
    name: str
    tier: str
    country: str
    age_days: int
    chargebacks: int = 0


@dataclass(frozen=True, slots=True)
class SeedTxn:
    id: str
    customer_id: str
    amount: float
    ago: timedelta
    descriptor: str
    status: str = "SUCCEEDED"
    gateway_status: str = "CAPTURED"
    card_last4: str = "4242"
    ip_country: str = "US"
    refunded_amount: float = 0.0


CUSTOMERS: tuple[SeedCustomer, ...] = (
    SeedCustomer("cus_alice001", "alice@acme-corp.com", "Alice Moreno", "ENTERPRISE", "US", 730),
    SeedCustomer("cus_bob00001", "bob@startup.io", "Bob Chen", "PRO", "US", 400),
    SeedCustomer("cus_carol001", "carol@gmail.com", "Carol Diaz", "FREE", "US", 365),
    SeedCustomer("cus_dave0001", "dave@example.com", "Dave Kim", "PRO", "CA", 200),
    SeedCustomer("cus_erin0001", "erin@designhub.co", "Erin Walsh", "PRO", "GB", 500),
    SeedCustomer("cus_frank001", "frank@oldmail.net", "Frank Ruiz", "FREE", "US", 900),
    SeedCustomer("cus_grace001", "grace@lumen.io", "Grace Park", "ENTERPRISE", "US", 1200),
    SeedCustomer("cus_mallory1", "mallory@protonmail.com", "Mallory Stone", "FREE", "US", 3, chargebacks=3),
    SeedCustomer("cus_henry001", "henry@bigretail.com", "Henry Okafor", "ENTERPRISE", "US", 1500),
)

_D = timedelta(days=1)
_M = timedelta(minutes=1)

TRANSACTIONS: tuple[SeedTxn, ...] = (
    # Alice: duplicate charge 90 seconds apart, $149 (auto-approvable)
    SeedTxn("txn_alice0001", "cus_alice001", 149.00, 3 * _D, "SENTINEL*CLOUD PRO PLAN"),
    SeedTxn("txn_alice0002", "cus_alice001", 149.00, 3 * _D - timedelta(seconds=90), "SENTINEL*CLOUD PRO PLAN"),
    SeedTxn("txn_alice0003", "cus_alice001", 49.00, 35 * _D, "SENTINEL*ADDON SEATS"),
    # Bob: duplicate charge $480 (requires HITL > $200)
    SeedTxn("txn_bob00001", "cus_bob00001", 480.00, 2 * _D, "SENTINEL*TEAM ANNUAL"),
    SeedTxn("txn_bob00002", "cus_bob00001", 480.00, 2 * _D - 2 * _M, "SENTINEL*TEAM ANNUAL"),
    # Carol: fraud — foreign IP, unknown card
    SeedTxn("txn_carol001", "cus_carol001", 12.00, 20 * _D, "SENTINEL*STARTER"),
    SeedTxn("txn_carol002", "cus_carol001", 899.99, 2 * _D, "SENTINEL*HARDWARE KIT", card_last4="9921", ip_country="RU"),
    # Dave: gateway timeout but ledger captured, then customer retried 25 min later
    SeedTxn("txn_dave0001", "cus_dave0001", 59.00, 1 * _D, "SENTINEL*PRO MONTHLY", gateway_status="TIMEOUT", ip_country="CA"),
    SeedTxn("txn_dave0002", "cus_dave0001", 59.00, 1 * _D - 25 * _M, "SENTINEL*PRO MONTHLY", ip_country="CA"),
    # Erin: refund within PRO window
    SeedTxn("txn_erin0001", "cus_erin0001", 29.00, 10 * _D, "SENTINEL*PRO MONTHLY", ip_country="GB"),
    # Frank: refund outside FREE window
    SeedTxn("txn_frank001", "cus_frank001", 99.00, 75 * _D, "SENTINEL*STARTER YEARLY"),
    # Grace: already refunded
    SeedTxn("txn_grace001", "cus_grace001", 350.00, 20 * _D, "SENTINEL*ENTERPRISE SEATS", status="REFUNDED", refunded_amount=350.00),
    # Mallory: new account, many chargebacks; used for prompt-injection scenarios
    SeedTxn("txn_mallo001", "cus_mallory1", 19.99, 2 * _D, "SENTINEL*STARTER"),
    # Henry: large refund in ENTERPRISE window (HITL by amount)
    SeedTxn("txn_henry001", "cus_henry001", 1250.00, 15 * _D, "SENTINEL*ENTERPRISE ANNUAL"),
    # Henry: old charge outside the default 90-day lookback (exercises the Planner -> DBInspector re-query loop)
    SeedTxn("txn_henry000", "cus_henry001", 300.00, 200 * _D, "SENTINEL*ONBOARDING SERVICES"),
)


async def seed_database(db: Database, *, force: bool = False, now: datetime | None = None) -> bool:
    """Insert seed rows. Returns True if seeding happened."""
    now = now or utcnow()
    async with db.connect() as conn:
        cur = await conn.execute("SELECT COUNT(*) FROM customers")
        row = await cur.fetchone()
        count = int(row[0]) if row else 0
        if count and not force:
            return False
        if force:
            for table in (
                "refunds", "card_freezes", "crm_notes", "email_outbox", "supervisor_alerts", "inbound_emails",
                "ticket_events", "tickets", "transactions", "customers",
            ):
                await conn.execute(f"DELETE FROM {table}")  # noqa: S608 - static table names
        await conn.executemany(
            "INSERT INTO customers (id, email, name, tier, country, account_created_at, chargeback_count, account_status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 'ACTIVE')",
            [(c.id, c.email, c.name, c.tier, c.country, iso(now - c.age_days * _D), c.chargebacks) for c in CUSTOMERS],
        )
        await conn.executemany(
            "INSERT INTO transactions (id, customer_id, amount, currency, status, gateway_status, created_at, "
            "merchant_descriptor, card_last4, ip_country, refunded_amount) VALUES (?, ?, ?, 'USD', ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    t.id,
                    t.customer_id,
                    t.amount,
                    t.status,
                    t.gateway_status,
                    iso(now - t.ago),
                    t.descriptor,
                    t.card_last4,
                    t.ip_country,
                    t.refunded_amount,
                )
                for t in TRANSACTIONS
            ],
        )
        await conn.commit()
    return True
