"""Deterministic fallback engine.

Used when no LLM is configured (``LLM_PROVIDER=mock``, CI) or when the provider is
degraded (circuit open, auth failure, schema-invalid output after repair). Every
LLM-backed node therefore has a safe, verifiable path — graceful degradation
instead of failure.
"""

from __future__ import annotations

import re

from app.agent.guardrails import (
    find_emails,
    find_money_amounts,
    find_transaction_refs,
    grounding_check,
)
from app.models.schemas import (
    ClaimCategory,
    CustomerDraft,
    JudgeVerdict,
    TransactionQuery,
    TriageExtraction,
    TXN_ID_PATTERN,
)

_CATEGORY_KEYWORDS: dict[ClaimCategory, tuple[str, ...]] = {
    "FRAUD_CLAIM": (
        r"\bfraud", r"\bunauthori[sz]ed", r"\bdid ?n[o']?t (make|authori[sz]e)", r"\bdon'?t recogni[sz]e",
        r"\bdo not recogni[sz]e", r"\bstolen\b", r"\bhacked\b", r"\bnot me\b", r"\bfraude\b", r"\bno reconozco\b",
    ),
    "DUPLICATE_CHARGE": (
        r"\btwice\b", r"\bdouble[- ]?charged?\b", r"\bduplicate", r"\bcharged (me )?(two|2) times\b", r"\btwo charges\b",
        r"\bcharged 2x\b", r"\bdos veces\b", r"\bdoble cobro\b", r"\bcobro duplicado\b",
    ),
    "REFUND_REQUEST": (
        r"\brefund", r"\bmoney back\b", r"\breimburse", r"\bcancel", r"\breembolso\b", r"\bdevoluci[oó]n\b",
        r"\bcharged\b.*\b(failed|error|timeout)", r"\b(failed|error|timeout)\b.*\bcharged\b",
    ),
}
_PRIORITY: tuple[ClaimCategory, ...] = ("FRAUD_CLAIM", "DUPLICATE_CHARGE", "REFUND_REQUEST")
_TXN_ID_RE = re.compile(TXN_ID_PATTERN)


def heuristic_triage(raw_message: str, provided_email: str | None) -> TriageExtraction:
    text = raw_message.lower()
    category: ClaimCategory = "GENERAL"
    for cat in _PRIORITY:
        if any(re.search(p, text) for p in _CATEGORY_KEYWORDS[cat]):
            category = cat
            break
    emails = find_emails(raw_message)
    amounts = [a for a in find_money_amounts(raw_message) if a <= 1_000_000]
    summary = re.sub(r"\s+", " ", raw_message).strip()[:400] or "(empty message)"
    return TriageExtraction(
        customer_email=provided_email or (emails[0] if emails else None),
        dispute_amount=amounts[0] if amounts else None,
        claim_category=category,
        transaction_refs=find_transaction_refs(raw_message),
        summary=summary,
    )


def normalise_transaction_ref(ref: str) -> str | None:
    """Repair common formatting mistakes: 'TXN-ALICE-0002' / 'txn #alice0002' -> 'txn_alice0002'."""
    body = re.sub(r"^txn[\s_\-#:]*", "", ref.strip(), flags=re.IGNORECASE)
    body = re.sub(r"[^A-Za-z0-9]", "", body).lower()
    candidate = f"txn_{body}"
    return candidate if _TXN_ID_RE.fullmatch(candidate) else None


def repair_query_args(raw_args: dict[str, object], customer_email: str | None) -> TransactionQuery | None:
    """Deterministic tool-argument repair used by the self-correction loop."""
    refs = raw_args.get("transaction_ids", [])
    fixed: list[str] = []
    if isinstance(refs, list):
        for r in refs:
            if isinstance(r, str) and (n := normalise_transaction_ref(r)) and n not in fixed:
                fixed.append(n)
    email = raw_args.get("customer_email") or customer_email
    lookback = raw_args.get("lookback_days", 90)
    try:
        return TransactionQuery(
            customer_email=email if isinstance(email, str) else None,
            transaction_ids=fixed,
            lookback_days=lookback if isinstance(lookback, int) and 1 <= lookback <= 365 else 90,
        )
    except ValueError:
        return None


def template_narrative(
    *,
    customer_line: str,
    findings: list[str],
    plan_line: str,
    risk_line: str,
    approval_line: str,
) -> str:
    bullet = "\n".join(f"- {f}" for f in findings) or "- No anomalies detected."
    return f"{customer_line}\nFindings:\n{bullet}\nProposed resolution: {plan_line}\nRisk: {risk_line}\n{approval_line}"


def template_draft(
    *,
    customer_name: str | None,
    ticket_id: str,
    action_type: str,
    executed_refunds: list[tuple[str, float]],
    customer_facing_reason: str,
    human_rejected: bool,
    execution_failed: bool,
) -> CustomerDraft:
    greeting = f"Hi {customer_name.split()[0]}," if customer_name else "Hello,"
    closing = "\n\nKind regards,\nSentinelOps Billing Support"
    ref = f"(reference {ticket_id})"
    if action_type == "REFUND" and executed_refunds and not execution_failed:
        total = round(sum(a for _, a in executed_refunds), 2)
        lines = "\n".join(f"- ${a:,.2f} USD for charge {tid}" for tid, a in executed_refunds)
        body = (
            f"{greeting}\n\nThank you for contacting us about your billing concern {ref}. {customer_facing_reason}\n\n"
            f"We have issued a refund totalling ${total:,.2f} USD:\n{lines}\n\n"
            "Card refunds usually settle within 5-10 business days, depending on your bank."
        )
        return CustomerDraft(subject="Your refund has been processed", body=body + closing)
    if human_rejected:
        body = (
            f"{greeting}\n\nThank you for your patience {ref}. Our billing specialists reviewed your request and were "
            "unable to approve a refund at this time. A member of our team will contact you within one business day "
            "to explain the outcome and the next steps."
        )
        return CustomerDraft(subject="Update on your billing request", body=body + closing)
    if execution_failed:
        body = (
            f"{greeting}\n\nThank you for contacting us {ref}. We approved your request but hit a temporary problem "
            "while processing it. A billing specialist is now handling it personally and will confirm once it is complete."
        )
        return CustomerDraft(subject="We are processing your request", body=body + closing)
    if action_type == "DENY":
        body = (
            f"{greeting}\n\nThank you for reaching out {ref}. We carefully reviewed your account. {customer_facing_reason} "
            "If you believe this is a mistake, reply to this email and your account manager will take a second look."
        )
        return CustomerDraft(subject="About your refund request", body=body + closing)
    if action_type == "REQUEST_INFO":
        body = (
            f"{greeting}\n\nThank you for contacting us {ref}. {customer_facing_reason} Please reply with the email "
            "address on your account and the charge reference shown on your statement so we can continue."
        )
        return CustomerDraft(subject="We need a few more details", body=body + closing)
    body = (
        f"{greeting}\n\nThank you for contacting us {ref}. {customer_facing_reason} "
        "A billing specialist will review your case and get back to you within one business day."
    )
    return CustomerDraft(subject="Your case is being reviewed", body=body + closing)


def heuristic_judge(
    draft: str,
    *,
    allowed_amounts: set[float],
    allowed_transaction_ids: set[str],
    required_amounts: set[float],
    threshold: float,
) -> JudgeVerdict:
    report = grounding_check(
        draft,
        allowed_amounts=allowed_amounts,
        allowed_transaction_ids=allowed_transaction_ids,
        required_amounts=required_amounts,
    )
    score = round(report.score, 2)
    return JudgeVerdict(
        passed=score >= threshold and not report.critical,
        faithfulness_score=score,
        feedback="Deterministic grounding audit: " + ("all statements supported." if not report.issues else "; ".join(report.issues)),
        issues=report.issues,
    )


def amounts_in(text: str) -> list[float]:
    return find_money_amounts(text)
