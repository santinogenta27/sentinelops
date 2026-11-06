"""Email-client-safe HTML templates (table layout, inline styles, every dynamic value escaped)."""

from __future__ import annotations

import html
import re

_BRAND = "#4f46e5"


def _paragraphs(text: str) -> str:
    """Plain text → HTML: blank-line paragraphs, '- ' bullet lists, line breaks. All escaped."""
    blocks: list[str] = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = [ln.rstrip() for ln in block.split("\n") if ln.strip()]
        if not lines:
            continue
        bullets = [ln for ln in lines if ln.lstrip().startswith("- ")]
        intro = [ln for ln in lines if not ln.lstrip().startswith("- ")]
        if intro:
            blocks.append(
                '<p style="margin:0 0 14px;line-height:1.6;">' + "<br>".join(html.escape(ln) for ln in intro) + "</p>"
            )
        if bullets:
            items = "".join(
                f'<li style="margin:0 0 6px;">{html.escape(ln.lstrip()[2:])}</li>' for ln in bullets
            )
            blocks.append(f'<ul style="margin:0 0 14px;padding-left:20px;line-height:1.6;">{items}</ul>')
    return "\n".join(blocks)


def _shell(*, preheader: str, header_label: str, header_color: str, inner: str, footer: str) -> str:
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="margin:0;padding:0;background:#f4f5f7;">
<span style="display:none;max-height:0;overflow:hidden;">{html.escape(preheader)}</span>
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#f4f5f7;padding:24px 12px;">
<tr><td align="center">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="max-width:600px;background:#ffffff;border-radius:12px;overflow:hidden;border:1px solid #e5e7eb;font-family:-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;color:#1f2937;font-size:15px;">
<tr><td style="background:{header_color};padding:18px 28px;color:#ffffff;">
  <div style="font-size:12px;letter-spacing:.08em;text-transform:uppercase;opacity:.85;">SentinelOps</div>
  <div style="font-size:18px;font-weight:700;margin-top:2px;">{html.escape(header_label)}</div>
</td></tr>
<tr><td style="padding:26px 28px 8px;">{inner}</td></tr>
<tr><td style="padding:14px 28px 22px;border-top:1px solid #f0f1f3;color:#6b7280;font-size:12px;line-height:1.5;">{footer}</td></tr>
</table>
</td></tr></table>
</body></html>"""


def render_customer_email(*, heading: str, body: str, ticket_id: str, support_address: str) -> str:
    footer = (
        f"Reference <b>{html.escape(ticket_id)}</b> · Reply to this email to continue the conversation.<br>"
        f"SentinelOps Billing Support · {html.escape(support_address)}"
    )
    return _shell(
        preheader=heading,
        header_label=heading,
        header_color=_BRAND,
        inner=_paragraphs(body),
        footer=footer,
    )


def render_supervisor_alert(
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
    cockpit_url: str,
    review_only: bool = False,
) -> str:
    rows = [
        ("Ticket", ticket_id),
        ("Requested action", f"{action_type} {amount:,.2f} {currency}"),
        ("Risk", risk_level),
        ("Channel", channel + (f" · from {sender}" if sender else "")),
    ]
    if subject:
        rows.append(("Subject", subject))
    table = "".join(
        f'<tr><td style="padding:6px 12px 6px 0;color:#6b7280;white-space:nowrap;vertical-align:top;">{html.escape(k)}</td>'
        f'<td style="padding:6px 0;font-weight:600;">{html.escape(v)}</td></tr>'
        for k, v in rows
    )
    flags = "".join(
        f'<span style="display:inline-block;margin:0 6px 6px 0;padding:2px 8px;border-radius:999px;background:#fef3c7;color:#92400e;font-size:12px;">{html.escape(f)}</span>'
        for f in risk_flags
    )
    intro = (
        "<b>🔎 A customer case needs a person.</b> The agent took no financial action and told the customer a specialist will follow up."
        if review_only
        else "<b>⚠️ A new customer email requires billing supervisor approval.</b> "
        "The agent paused execution at the Human-in-the-Loop gate; no money has moved."
    )
    inner = (
        f'<p style="margin:0 0 14px;line-height:1.6;">{intro}</p>'
        f'<table role="presentation" cellpadding="0" cellspacing="0" style="margin:0 0 14px;font-size:14px;">{table}</table>'
        f'<div style="margin:0 0 12px;">{flags}</div>'
        f'<div style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:8px;padding:12px;font-size:13px;line-height:1.55;margin:0 0 18px;">{_paragraphs(summary)}</div>'
        f'<a href="{html.escape(cockpit_url)}" style="display:inline-block;background:#d97706;color:#ffffff;text-decoration:none;'
        'padding:10px 18px;border-radius:8px;font-weight:700;">Review in SentinelOps Cockpit</a>'
    )
    return _shell(
        preheader=f"{'Review' if review_only else 'Approval'} needed: {ticket_id}",
        header_label="Review needed" if review_only else "Approval required",
        header_color="#d97706",
        inner=inner,
        footer="Automated alert · SentinelOps Human-in-the-Loop gate",
    )
