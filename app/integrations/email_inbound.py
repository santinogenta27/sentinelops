"""Inbound email normalisation.

Accepts the common inbound-parse payload shapes (generic JSON ``from/subject/body_plain/message_id``,
Mailgun ``sender/body-plain/Message-Id``, SendGrid ``from/text``, Postmark ``From/TextBody/MessageID``)
and raw RFC 822 messages fetched over IMAP, and turns them into one strict ``InboundEmail``.
"""

from __future__ import annotations

import hashlib
import html
import re
from email import message_from_bytes, policy
from email.message import EmailMessage
from email.utils import parseaddr
from typing import Annotated, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, StringConstraints

from app.models.schemas import StrictModel, normalise_email

InboundSource = Literal["webhook", "imap", "simulator"]
MAX_BODY_CHARS = 5000

_QUOTE_HEADER_RE = re.compile(
    r"^(On .{3,200} wrote:|El .{3,200} escribi[oó]:|-{2,}\s*Original Message\s*-{2,}|_{5,}|From: .+ Sent: .+)\s*$",
    re.IGNORECASE,
)
_MOBILE_SIG_RE = re.compile(r"^(Sent from my .+|Enviado desde mi .+|Get Outlook for .+)$", re.IGNORECASE)
_TAG_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>|<[^>]+>", re.IGNORECASE | re.DOTALL)


class InboundEmailPayload(BaseModel):
    """Tolerant webhook payload (providers send many extra fields, so extras are ignored)."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True, str_strip_whitespace=True)

    from_: str = Field(validation_alias=AliasChoices("from", "from_", "sender", "From", "from_address"))
    subject: str = Field(default="", max_length=500, validation_alias=AliasChoices("subject", "Subject"))
    body_plain: str = Field(
        default="",
        max_length=200_000,
        validation_alias=AliasChoices("body_plain", "body-plain", "stripped-text", "text", "TextBody", "body"),
    )
    message_id: str | None = Field(
        default=None,
        max_length=500,
        validation_alias=AliasChoices("message_id", "Message-Id", "Message-ID", "message-id", "MessageID"),
    )
    source: Literal["webhook", "simulator"] = "webhook"


class InboundEmail(StrictModel):
    from_address: str
    from_name: Annotated[str, StringConstraints(max_length=200)] = ""
    subject: Annotated[str, StringConstraints(max_length=500)] = ""
    body: Annotated[str, StringConstraints(max_length=MAX_BODY_CHARS)]
    message_id: Annotated[str, StringConstraints(min_length=3, max_length=500)]
    source: InboundSource

    def ticket_text(self) -> str:
        """What the agent reads: subject + cleaned body (the From header is identity, not content)."""
        subject = self.subject or "(no subject)"
        return f"Subject: {subject}\n\n{self.body}"[:6000]


def html_to_text(markup: str) -> str:
    text = re.sub(r"<br\s*/?>|</p>|</div>|</li>", "\n", markup, flags=re.IGNORECASE)
    return html.unescape(_TAG_RE.sub("", text))


def clean_email_body(text: str) -> str:
    """Strip quoted replies, signatures and noise so the agent only sees the new message."""
    lines: list[str] = []
    for raw in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw.rstrip()
        if _QUOTE_HEADER_RE.match(line.strip()) or line == "--":  # RFC 3676 signature delimiter "-- "
            break
        if line.lstrip().startswith(">") or _MOBILE_SIG_RE.match(line.strip()):
            continue
        lines.append(line)
    body = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    return body[:MAX_BODY_CHARS]


def synthetic_message_id(from_address: str, subject: str, body: str) -> str:
    """Deterministic id for payloads without Message-Id, so provider retries stay idempotent."""
    digest = hashlib.sha256(f"{from_address}\x00{subject}\x00{body}".encode()).hexdigest()[:24]
    return f"<gen-{digest}@sentinelops>"


def build_inbound_email(
    *, raw_from: str, subject: str, body: str, message_id: str | None, source: InboundSource
) -> InboundEmail:
    name, address = parseaddr(raw_from)
    email_address = normalise_email(address or raw_from)
    if email_address is None:
        raise ValueError("sender address is missing")
    clean_subject = re.sub(r"\s+", " ", subject).strip()[:500]
    clean_body = clean_email_body(body)
    if not clean_body and not clean_subject:
        raise ValueError("email has neither subject nor body")
    return InboundEmail(
        from_address=email_address,
        from_name=name.strip()[:200],
        subject=clean_subject,
        body=clean_body or clean_subject,
        message_id=(message_id or "").strip() or synthetic_message_id(email_address, clean_subject, clean_body),
        source=source,
    )


def from_webhook_payload(payload: InboundEmailPayload) -> InboundEmail:
    return build_inbound_email(
        raw_from=payload.from_,
        subject=payload.subject,
        body=payload.body_plain,
        message_id=payload.message_id,
        source=payload.source,
    )


def is_auto_generated(msg: EmailMessage) -> bool:
    """Never answer bounces / auto-replies (prevents mail loops)."""
    auto = str(msg.get("Auto-Submitted", "no")).lower()
    sender = str(msg.get("From", "")).lower()
    return auto != "no" or "mailer-daemon" in sender or str(msg.get("Precedence", "")).lower() in {"bulk", "junk", "list"}


def parse_rfc822(raw: bytes) -> tuple[EmailMessage, InboundEmail]:
    msg = message_from_bytes(raw, policy=policy.default)
    if not isinstance(msg, EmailMessage):  # pragma: no cover - policy.default always yields EmailMessage
        raise ValueError("unparseable message")
    part = msg.get_body(preferencelist=("plain", "html"))
    body = ""
    if part is not None:
        content = part.get_content()
        body = content if isinstance(content, str) else ""
        if part.get_content_subtype() == "html":
            body = html_to_text(body)
    email = build_inbound_email(
        raw_from=str(msg.get("From", "")),
        subject=str(msg.get("Subject", "")),
        body=body,
        message_id=str(msg.get("Message-ID", "")) or None,
        source="imap",
    )
    return msg, email
