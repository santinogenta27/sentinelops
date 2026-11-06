"""Inbound email → ticket. Shared by the webhook endpoint, the IMAP listener and the cockpit simulator."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.integrations.email_inbound import InboundEmail, is_own_outbound
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
        # Self-send is allowed (demo: write from the support mailbox to itself); only the replies
        # this engine sent are refused, so answering ourselves can never loop.
        if is_own_outbound(email.message_id):
            raise ValueError("refusing to ingest a reply sent by SentinelOps itself (loop protection)")
        tid = new_ticket_id()
        # Looked up before this email is recorded. Only a reply from the same sender continues a case.
        parent = await rt.db.find_clarification(email.from_address, email.thread_ids)
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
        alias_of: str | None = None
        if customer is None:
            customer = await rt.deps.repo.get_customer_by_alias(email.from_address)
            alias_of = customer.email if customer else None
        meta = ChannelMeta(
            channel="email",
            inbound_source=email.source,
            sender_email=email.from_address,
            subject=email.subject or None,
            message_id=email.message_id,
            sender_alias_of=alias_of,
            identity_unverified=email.sender_auth_failed,
            parent_ticket_id=str(parent["ticket_id"]) if parent else None,
            clarification_round=int(str(parent["clarification_round"])) + 1 if parent else 0,
        )
        text = email.ticket_text()
        if parent:
            # What this customer wrote earlier in this thread, and nothing from any other ticket.
            text = f"{text}\n\n[Earlier message from this customer in this thread]\n{parent['raw_message']}"[:6000]
            await rt.db.mark_case_continued(str(parent["ticket_id"]), continued_by=tid)
            await rt.emit(str(parent["ticket_id"]), "clarification_received", None, {"continued_by": tid})
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
                "verified_alias_of": alias_of,
            },
        )
        customer_email = customer.email if customer else None
        if wait:
            await rt.run_ticket(text, customer_email, ticket_id=tid, meta=meta)
        else:
            rt.submit(text, customer_email, ticket_id=tid, meta=meta)
        return IngestResult(tid, False, email.message_id, customer is not None)
