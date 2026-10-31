"""Inbound email → ticket. Shared by the webhook endpoint, the IMAP listener and the cockpit simulator."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.integrations.email_inbound import InboundEmail
from app.models.schemas import ChannelMeta
from app.services.runtime import SentinelRuntime, new_ticket_id

logger = logging.getLogger("sentinelops.inbound")


@dataclass(frozen=True, slots=True)
class IngestResult:
    ticket_id: str
    duplicate: bool
    message_id: str
    sender_is_customer: bool


class InboundEmailService:
    def __init__(self, runtime: SentinelRuntime) -> None:
        self.runtime = runtime

    async def ingest(self, email: InboundEmail, *, wait: bool = False) -> IngestResult:
        """Create (exactly once per Message-Id) a ticket from an email and start the agent.

        Identity: if the sender owns an account, that address is the authenticated customer
        identity. Otherwise the agent identifies the account from the body, and the planner
        treats the request as coming from an unverified sender (forced HITL on payouts).
        """
        rt = self.runtime
        if email.from_address == rt.settings.support_email_address.lower():
            raise ValueError("refusing to ingest mail sent by our own support address (loop protection)")
        tid = new_ticket_id()
        existing = await rt.db.record_inbound_email(
            message_id=email.message_id,
            ticket_id=tid,
            from_address=email.from_address,
            from_name=email.from_name,
            subject=email.subject,
            body=email.body,
            source=email.source,
        )
        if existing is not None:
            logger.info("duplicate inbound email %s → %s", email.message_id, existing)
            return IngestResult(existing, True, email.message_id, False)

        customer = await rt.deps.repo.get_customer_by_email(email.from_address)
        meta = ChannelMeta(
            channel="email",
            inbound_source=email.source,
            sender_email=email.from_address,
            subject=email.subject or None,
            message_id=email.message_id,
        )
        await rt.emit(
            tid,
            "email_received",
            None,
            {
                "from": email.from_address,
                "from_name": email.from_name,
                "subject": email.subject,
                "source": email.source,
                "message_id": email.message_id,
                "known_customer": customer is not None,
            },
        )
        customer_email = customer.email if customer else None
        if wait:
            await rt.run_ticket(email.ticket_text(), customer_email, ticket_id=tid, meta=meta)
        else:
            rt.submit(email.ticket_text(), customer_email, ticket_id=tid, meta=meta)
        return IngestResult(tid, False, email.message_id, customer is not None)
