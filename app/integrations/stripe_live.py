"""Real Stripe integration (stripe-python): live dashboard + TEST-mode mirror of ledger refunds.

The embedded ledger (``StripeMock``) stays the system of record for every money decision.
When a Stripe key is configured, each ledger refund is *mirrored* into Stripe so operators
can watch the money move in the official dashboard:

* the original charge of a ledger transaction is created once in Stripe (test card
  ``pm_card_bypassPending`` so the funds are immediately available), then
* a Stripe ``Refund`` is created against it with an idempotency key derived from the ledger
  refund id — retries and graph replays can never refund twice.

Safety: writes are only ever performed with a TEST key (``sk_test_`` / ``rk_test_``). A live key
is accepted for the read-only dashboard, but mirroring is refused — this service never creates
real charges or refunds. Mirror failures never fail a ticket; they are recorded and surfaced.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

from app.config import Settings
from app.db.database import Database, iso, utcnow

logger = logging.getLogger("sentinelops.stripe")

StripeMode = Literal["disabled", "test", "live", "invalid"]
MirrorStatus = Literal["mirrored", "duplicate", "skipped", "failed"]
TEST_PAYMENT_METHOD = "pm_card_bypassPending"  # Stripe test card: funds land in the available balance
ZERO_DECIMAL = frozenset({"BIF", "CLP", "DJF", "GNF", "JPY", "KMF", "KRW", "MGA", "PYG", "RWF", "UGX", "VND", "VUV", "XAF", "XOF", "XPF"})


def key_mode(key: str | None) -> StripeMode:
    if not key:
        return "disabled"
    if key.startswith(("sk_test_", "rk_test_")):
        return "test"
    if key.startswith(("sk_live_", "rk_live_")):
        return "live"
    return "invalid"


def to_minor(amount: float, currency: str) -> int:
    return int(round(amount)) if currency.upper() in ZERO_DECIMAL else int(round(amount * 100))


def to_major(minor: int, currency: str) -> float:
    return float(minor) if currency.upper() in ZERO_DECIMAL else round(minor / 100, 2)


def refund_reason(reason_code: str) -> str:
    code = reason_code.upper()
    if "DUPLICATE" in code:
        return "duplicate"
    if "FRAUD" in code:
        return "fraudulent"
    return "requested_by_customer"


class StripeBackend(Protocol):
    """The narrow, blocking slice of the Stripe API we use (fakeable in tests). Returns plain dicts."""

    def account_id(self) -> str: ...
    def balance(self) -> dict[str, Any]: ...
    def list_charges(self, limit: int) -> list[dict[str, Any]]: ...
    def list_refunds(self, limit: int) -> list[dict[str, Any]]: ...
    def create_payment(
        self, *, amount: int, currency: str, description: str, metadata: dict[str, str], idempotency_key: str
    ) -> dict[str, Any]: ...
    def create_refund(
        self, *, payment_intent: str, amount: int, reason: str, metadata: dict[str, str], idempotency_key: str
    ) -> dict[str, Any]: ...


class StripeSdkBackend:
    """``StripeBackend`` over the official stripe-python ``StripeClient``."""

    def __init__(self, api_key: str) -> None:
        import stripe  # local import: the module is optional at runtime

        self._client = stripe.StripeClient(api_key, max_network_retries=2)

    @staticmethod
    def _plain(obj: Any) -> dict[str, Any]:
        data = obj.to_dict()
        return data if isinstance(data, dict) else {}

    def account_id(self) -> str:
        return str(self._client.v1.accounts.retrieve_current().id)

    def balance(self) -> dict[str, Any]:
        return self._plain(self._client.v1.balance.retrieve())

    def list_charges(self, limit: int) -> list[dict[str, Any]]:
        return [self._plain(c) for c in self._client.v1.charges.list(params={"limit": limit}).data]

    def list_refunds(self, limit: int) -> list[dict[str, Any]]:
        return [self._plain(r) for r in self._client.v1.refunds.list(params={"limit": limit}).data]

    def create_payment(
        self, *, amount: int, currency: str, description: str, metadata: dict[str, str], idempotency_key: str
    ) -> dict[str, Any]:
        pi = self._client.v1.payment_intents.create(
            params={
                "amount": amount,
                "currency": currency.lower(),
                "payment_method": TEST_PAYMENT_METHOD,
                "automatic_payment_methods": {"enabled": True, "allow_redirects": "never"},
                "confirm": True,
                "description": description,
                "metadata": metadata,
            },
            options={"idempotency_key": idempotency_key},
        )
        return self._plain(pi)

    def create_refund(
        self, *, payment_intent: str, amount: int, reason: str, metadata: dict[str, str], idempotency_key: str
    ) -> dict[str, Any]:
        refund = self._client.v1.refunds.create(
            params={
                "payment_intent": payment_intent,
                "amount": amount,
                "reason": reason,  # type: ignore[typeddict-item]
                "metadata": metadata,
            },
            options={"idempotency_key": idempotency_key},
        )
        return self._plain(refund)


@dataclass(frozen=True, slots=True)
class MirrorResult:
    status: MirrorStatus
    payment_intent_id: str | None = None
    stripe_refund_id: str | None = None
    dashboard_url: str | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _ts(epoch: object) -> str | None:
    return datetime.fromtimestamp(int(epoch), UTC).isoformat() if isinstance(epoch, (int, float)) else None


class StripeLive:
    MIRROR_TOOL = "stripe.live_mirror"

    def __init__(
        self,
        db: Database,
        settings: Settings,
        backend_factory: Callable[[str], StripeBackend] | None = None,
    ) -> None:
        self.db = db
        self.settings = settings
        self._factory = backend_factory or StripeSdkBackend
        self._backend: StripeBackend | None = None
        self._account: str | None = None
        self._cache: tuple[float, dict[str, Any]] | None = None
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------ status
    @property
    def mode(self) -> StripeMode:
        return key_mode(self.settings.stripe_key)

    @property
    def mirror_enabled(self) -> bool:
        return self.mode == "test" and self.settings.stripe_mirror_refunds

    def _client(self) -> StripeBackend:
        key = self.settings.stripe_key
        if key is None:
            raise RuntimeError("STRIPE_API_KEY is not configured")
        if self._backend is None:
            self._backend = self._factory(key)
        return self._backend

    async def _account_id(self) -> str:
        if self._account is None:
            self._account = await asyncio.to_thread(self._client().account_id)
        return self._account

    def _dashboard(self, path: str) -> str:
        acct = f"{self._account}/" if self._account else ""
        test = "test/" if self.mode == "test" else ""
        return f"https://dashboard.stripe.com/{acct}{test}{path}"

    def describe(self) -> dict[str, object]:
        return {
            "configured": self.mode in ("test", "live"),
            "mode": self.mode,
            "mirror_enabled": self.mirror_enabled,
            "account_id": self._account,
        }

    # ------------------------------------------------------------------ mirror
    async def mirror_refund(
        self,
        *,
        local_refund_id: str,
        ticket_id: str,
        transaction_id: str,
        transaction_amount: float,
        amount: float,
        currency: str,
        reason_code: str,
        customer_email: str | None,
        customer_name: str | None,
        approved_by: str | None,
    ) -> MirrorResult:
        """Mirror one ledger refund into Stripe TEST mode. Idempotent per ``local_refund_id``."""
        if not self.mirror_enabled:
            return MirrorResult("skipped", error=f"mirror disabled (stripe mode: {self.mode})")
        existing = await self.db.get_stripe_mirror(local_refund_id)
        if existing is not None and existing["status"] == "mirrored":
            pi = str(existing["payment_intent_id"] or "")
            return MirrorResult(
                "duplicate", pi or None, str(existing["stripe_refund_id"] or "") or None,
                self._dashboard(f"payments/{pi}") if pi else None,
            )
        async with self._lock:  # serialise charge creation per process (one Stripe charge per txn)
            try:
                await self._account_id()
                client = self._client()
                charge = await self.db.get_stripe_charge(transaction_id)
                if charge is None:
                    meta = {"source": "sentinelops", "ledger_transaction_id": transaction_id,
                            "customer_email": customer_email or "", "customer_name": customer_name or ""}
                    created = await asyncio.to_thread(
                        lambda: client.create_payment(
                            amount=to_minor(transaction_amount, currency),
                            currency=currency,
                            description=f"SentinelOps ledger charge {transaction_id} · {customer_email or 'unknown'}",
                            metadata=meta,
                            idempotency_key=f"sentinelops-charge-{transaction_id}-{local_refund_id}",
                        )
                    )
                    if created.get("status") != "succeeded":
                        raise RuntimeError(f"test charge not captured (status={created.get('status')})")
                    latest = created.get("latest_charge")
                    await self.db.save_stripe_charge(
                        transaction_id=transaction_id,
                        payment_intent_id=str(created["id"]),
                        charge_id=str(latest) if isinstance(latest, str) else None,
                        amount_cents=to_minor(transaction_amount, currency),
                        currency=currency,
                    )
                    payment_intent = str(created["id"])
                else:
                    payment_intent = str(charge["payment_intent_id"])
                refund = await asyncio.to_thread(
                    lambda: client.create_refund(
                        payment_intent=payment_intent,
                        amount=to_minor(amount, currency),
                        reason=refund_reason(reason_code),
                        metadata={
                            "source": "sentinelops", "ticket_id": ticket_id, "ledger_refund_id": local_refund_id,
                            "ledger_transaction_id": transaction_id, "customer_email": customer_email or "",
                            "customer_name": customer_name or "", "approved_by": approved_by or "",
                        },
                        idempotency_key=f"sentinelops-refund-{local_refund_id}",
                    )
                )
                result = MirrorResult("mirrored", payment_intent, str(refund["id"]), self._dashboard(f"payments/{payment_intent}"))
            except Exception as exc:  # noqa: BLE001 - mirroring must never break the ticket
                message = f"{type(exc).__name__}: {str(exc)[:240]}"
                logger.warning("stripe mirror failed for %s/%s: %s", ticket_id, local_refund_id, message)
                result = MirrorResult("failed", error=message)
        await self.db.save_stripe_mirror(
            local_refund_id=local_refund_id, ticket_id=ticket_id, transaction_id=transaction_id, amount=amount,
            currency=currency, status=result.status, payment_intent_id=result.payment_intent_id,
            stripe_refund_id=result.stripe_refund_id, error=result.error,
        )
        self._cache = None
        return result

    # --------------------------------------------------------------- dashboard
    async def dashboard(self, limit: int = 15, *, max_age_s: float = 2.0) -> dict[str, Any]:
        """Balance + latest charges/refunds straight from Stripe (short cache to absorb UI bursts)."""
        base: dict[str, Any] = {**self.describe(), "fetched_at": iso(utcnow()), "error": None,
                                "balance": None, "charges": [], "refunds": [],
                                "mirror": await self.db.list_stripe_mirror(limit)}
        if self.mode not in ("test", "live"):
            base["error"] = None if self.mode == "disabled" else "STRIPE_API_KEY is not a valid secret/restricted key"
            return base
        if self._cache is not None and time.monotonic() - self._cache[0] < max_age_s:
            return {**self._cache[1], "mirror": base["mirror"]}
        try:
            await self._account_id()
            client = self._client()
            balance, charges, refunds = await asyncio.gather(
                asyncio.to_thread(client.balance),
                asyncio.to_thread(client.list_charges, limit),
                asyncio.to_thread(client.list_refunds, limit),
            )
        except Exception as exc:  # noqa: BLE001 - surface provider errors to the cockpit
            base["error"] = f"{type(exc).__name__}: {str(exc)[:240]}"
            return base

        by_refund = {str(m["stripe_refund_id"]): m for m in base["mirror"] if m.get("stripe_refund_id")}
        base.update(self.describe())

        def money_list(items: object) -> list[dict[str, Any]]:
            out: list[dict[str, Any]] = []
            for b in items if isinstance(items, list) else []:
                cur = str(b.get("currency", "usd")).upper()
                out.append({"currency": cur, "amount": to_major(int(b.get("amount", 0)), cur)})
            return out

        base["balance"] = {"available": money_list(balance.get("available")), "pending": money_list(balance.get("pending"))}
        base["charges"] = [
            {
                "id": c.get("id"),
                "amount": to_major(int(c.get("amount", 0)), str(c.get("currency", "usd"))),
                "amount_refunded": to_major(int(c.get("amount_refunded", 0)), str(c.get("currency", "usd"))),
                "currency": str(c.get("currency", "usd")).upper(),
                "status": c.get("status"),
                "refunded": bool(c.get("refunded")),
                "created": _ts(c.get("created")),
                "description": c.get("description"),
                "customer": (c.get("metadata") or {}).get("customer_email") or (c.get("billing_details") or {}).get("email"),
                "receipt_url": c.get("receipt_url"),
                "dashboard_url": self._dashboard(f"payments/{c.get('payment_intent') or c.get('id')}"),
            }
            for c in charges
        ]
        base["refunds"] = [
            {
                "id": r.get("id"),
                "amount": to_major(int(r.get("amount", 0)), str(r.get("currency", "usd"))),
                "currency": str(r.get("currency", "usd")).upper(),
                "status": r.get("status"),
                "reason": r.get("reason"),
                "created": _ts(r.get("created")),
                "charge": r.get("charge"),
                "payment_intent": r.get("payment_intent"),
                "ticket_id": (r.get("metadata") or {}).get("ticket_id")
                or (by_refund.get(str(r.get("id"))) or {}).get("ticket_id"),
                "customer": (r.get("metadata") or {}).get("customer_email"),
                "approved_by": (r.get("metadata") or {}).get("approved_by"),
                "dashboard_url": self._dashboard(f"payments/{r.get('payment_intent') or r.get('charge')}"),
            }
            for r in refunds
        ]
        self._cache = (time.monotonic(), base)
        return base
