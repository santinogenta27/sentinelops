"""Outbound mail: customer replies + supervisor alerts.

Every message is persisted to the outbox (with its rendered HTML) idempotently per ticket.
Real SMTP delivery is optional (``ENABLE_EMAIL_SENDER=true``) and deliberately narrow:
customer replies are only delivered for tickets that genuinely arrived by email (IMAP or a
provider webhook) — never for cockpit simulations or web tickets about seed customers, so the
demo can never email real third-party domains by accident.
"""

from __future__ import annotations

import asyncio
import logging
import smtplib
import ssl
from collections.abc import Callable
from email.message import EmailMessage
from typing import Literal

from app.config import Settings
from app.db.database import Database, iso, utcnow
from app.integrations.email_templates import render_customer_email, render_supervisor_alert
from app.integrations.faults import FaultInjector

logger = logging.getLogger("sentinelops.mail")

DeliveryStatus = Literal["smtp", "outbox", "duplicate", "smtp_failed"]
Transport = Callable[[EmailMessage], None]
REAL_DELIVERY_SOURCES = frozenset({"imap", "webhook"})


class Mailer:
    TOOL_NAME = "email.send"

    def __init__(
        self, db: Database, faults: FaultInjector, settings: Settings, transport: Transport | None = None
    ) -> None:
        self.db = db
        self.faults = faults
        self.settings = settings
        self._transport = transport

    # ------------------------------------------------------------------ smtp
    @property
    def real_delivery_enabled(self) -> bool:
        return self.settings.enable_email_sender and (
            self._transport is not None or self.settings.smtp_credentials is not None
        )

    def _smtp_send(self, msg: EmailMessage) -> None:
        creds = self.settings.smtp_credentials
        if creds is None:
            raise RuntimeError("SMTP credentials are not configured")
        with smtplib.SMTP(self.settings.smtp_host, self.settings.smtp_port, timeout=20) as smtp:
            smtp.starttls(context=ssl.create_default_context())
            smtp.login(*creds)
            smtp.send_message(msg)

    def _from_header(self) -> str:
        creds = self.settings.smtp_credentials
        address = creds[0] if creds else self.settings.support_email_address
        return f"SentinelOps Billing <{address}>"

    async def _deliver(self, msg: EmailMessage) -> None:
        transport = self._transport or self._smtp_send
        await asyncio.to_thread(transport, msg)

    @staticmethod
    def _build(*, sender: str, to: str, subject: str, text: str, html_body: str, in_reply_to: str | None) -> EmailMessage:
        msg = EmailMessage()
        msg["From"] = sender
        msg["To"] = to
        msg["Subject"] = subject
        if in_reply_to:
            msg["In-Reply-To"] = in_reply_to
            msg["References"] = in_reply_to
        msg.set_content(text)
        msg.add_alternative(html_body, subtype="html")
        return msg

    # ------------------------------------------------------------- customer
    async def send_customer_reply(
        self,
        *,
        ticket_id: str,
        to_address: str | None,
        subject: str,
        heading: str,
        body: str,
        channel: str,
        inbound_source: str | None,
        in_reply_to: str | None,
    ) -> DeliveryStatus:
        self.faults.check(self.TOOL_NAME)
        html_body = render_customer_email(
            heading=heading, body=body, ticket_id=ticket_id, support_address=self.settings.support_email_address
        )
        async with self.db.connect() as conn:
            cur = await conn.execute(
                "INSERT OR IGNORE INTO email_outbox (ticket_id, to_address, subject, body, body_html, delivery, channel, "
                "in_reply_to, created_at) VALUES (?, ?, ?, ?, ?, 'outbox', ?, ?, ?)",
                (ticket_id, to_address, subject, body, html_body, channel, in_reply_to, iso(utcnow())),
            )
            await conn.commit()
            if (cur.rowcount or 0) == 0:
                return "duplicate"  # resumed / replayed graph: never email the customer twice

        if not (
            self.real_delivery_enabled and channel == "email" and inbound_source in REAL_DELIVERY_SOURCES and to_address
        ):
            return "outbox"
        status: DeliveryStatus = "smtp"
        try:
            await self._deliver(
                self._build(
                    sender=self._from_header(), to=to_address, subject=subject, text=body,
                    html_body=html_body, in_reply_to=in_reply_to,
                )
            )
        except Exception as exc:  # noqa: BLE001 - delivery failure must not fail the ticket
            logger.warning("SMTP delivery failed for %s: %s", ticket_id, exc)
            status = "smtp_failed"
        async with self.db.connect() as conn:
            await conn.execute("UPDATE email_outbox SET delivery = ? WHERE ticket_id = ?", (status, ticket_id))
            await conn.commit()
        return status

    # ------------------------------------------------------------ supervisor
    async def send_supervisor_alert(
        self,
        *,
        ticket_id: str,
        amount: float,
        currency: str,
        action_type: str,
        risk_level: str,
        risk_flags: list[str],
        channel: str,
        sender: str | None,
        subject: str | None,
        summary: str,
        cockpit_url: str = "http://127.0.0.1:8000/",
    ) -> bool:
        """Persist (and optionally email) the HITL alert. Returns False if already alerted for this ticket."""
        alert_subject = f"⚠️ Approval required: {action_type} {amount:,.2f} {currency} · {ticket_id}"
        html_body = render_supervisor_alert(
            ticket_id=ticket_id, amount=amount, currency=currency, action_type=action_type, risk_level=risk_level,
            risk_flags=risk_flags, channel=channel, sender=sender, subject=subject, summary=summary,
            cockpit_url=cockpit_url,
        )
        recipient = self.settings.supervisor_email
        real = self.real_delivery_enabled and not recipient.endswith(".demo")
        async with self.db.connect() as conn:
            cur = await conn.execute(
                "INSERT OR IGNORE INTO supervisor_alerts (ticket_id, recipient, subject, body_html, amount, risk_level, "
                "channel, delivery, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (ticket_id, recipient, alert_subject, html_body, amount, risk_level, channel,
                 "pending" if real else "simulated", iso(utcnow())),
            )
            await conn.commit()
            if (cur.rowcount or 0) == 0:
                return False
        if real:
            delivery = "smtp"
            try:
                await self._deliver(
                    self._build(
                        sender=self._from_header(), to=recipient, subject=alert_subject,
                        text=f"Approval required for {ticket_id}: {action_type} {amount:,.2f} {currency}.\n\n{summary}",
                        html_body=html_body, in_reply_to=None,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("supervisor alert delivery failed for %s: %s", ticket_id, exc)
                delivery = "smtp_failed"
            async with self.db.connect() as conn:
                await conn.execute("UPDATE supervisor_alerts SET delivery = ? WHERE ticket_id = ?", (delivery, ticket_id))
                await conn.commit()
        return True
