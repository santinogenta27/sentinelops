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
        r"\bdo not recogni[sz]e", r"\bstolen\b", r"\bhacked\b", r"\bnot me\b", r"\bwasn'?t (me|us)\b", r"\bfraude\b",
        r"\bno reconozco\b", r"\bnever (made|bought|ordered|authori[sz]ed)\b", r"\bno (hice|autoric[eé]|compr[eé])\b",
        r"\bnunca compr[eé]\b", r"\bno autorizad[oa]\b", r"\bsin (mi|nuestra) (autorizaci[oó]n|permiso|aprobaci[oó]n)\b",
        r"\bwithout (my|our) (permission|authori[sz]ation|approval|consent)\b",
    ),
    "DUPLICATE_CHARGE": (
        r"\btwice\b", r"\bdouble[- ]?charged?\b", r"\bduplicate", r"\b(charged|billed) (me )?(two|2|three|3) times\b",
        r"\btwo charges\b", r"\bcharged 2x\b", r"\b(dos|2|tres|3) veces\b", r"\b(dos|2|tres|3) (cargos|cobros)\b", r"\bdoble cobro\b",
        r"\b(cobro|cargo) duplicado\b", r"\bdeux fois\b",
    ),
    # A bare "cancel" is deliberately not here: cancelling a subscription is not a refund request.
    "REFUND_REQUEST": (
        r"\brefund", r"\bmoney back\b", r"\breimburs", r"\breembols", r"\bdevoluci[oó]n\b", r"\bdevolv", r"\bdevuelv",
        r"\brembours", r"\b(want|need|get|give)\b[^.\n]{0,40}(\bmoney\b|\$\s?\d|\bcharge\b|\bpayment\b)[^.\n]{0,20}\bback\b",
        r"\breintegr", r"\bacredit", r"\bcredit\b[^.\n]{0,30}\bback\b",
        r"\bcharged\b.*\b(failed|error|timeout)", r"\b(failed|error|timeout)\b.*\bcharged\b",
    ),
}
_PRIORITY: tuple[ClaimCategory, ...] = ("FRAUD_CLAIM", "DUPLICATE_CHARGE", "REFUND_REQUEST")

# Explicit "give me my money back" intent (EN/ES/FR/PT). Deliberately excludes a bare "cancel":
# cancelling a subscription is not a refund request.
_REFUND_INTENT = re.compile(
    r"\brefund|\bmoney back\b|\breimburs|\breverse\b|\breembols|\bdevol[uv]|\bdevuelv|\brembours|\breembolso|\bestorno"
    r"|\breintegr|\bacredit|\bcredit\b[^.\n]{0,30}\bback\b"
    r"|\b(want|need|get|give)\b[^.\n]{0,40}(\bmoney\b|\$\s?\d|\bcharge\b|\bpayment\b)[^.\n]{0,20}\bback\b",
    re.IGNORECASE,
)


_PARTIAL_INTENT = re.compile(r"\bpartial|\bparcial|\bpart of\b|\bhalf (of|the|my|that|this)\b|\bla mitad\b|\bpro-?rat|\bprorrat", re.IGNORECASE)
_CANCEL_INTENT = re.compile(r"\bcancel|\bdar(me|nos)? de baja\b|\bbaja de (mi|la) suscripci|\bunsubscribe\b", re.IGNORECASE)


def requests_refund(raw_message: str) -> bool:
    """Did the customer explicitly ask for money back? Gates discretionary (in-window) refunds."""
    return bool(_REFUND_INTENT.search(raw_message))


def requests_partial_refund(raw_message: str) -> bool:
    return bool(_PARTIAL_INTENT.search(raw_message))


def requests_cancellation(raw_message: str) -> bool:
    return bool(_CANCEL_INTENT.search(raw_message))
_TXN_ID_RE = re.compile(TXN_ID_PATTERN)


_CARD_LAST4 = re.compile(
    r"(?:ending(?: in| with)?|ends? (?:in|with)|last (?:4|four)(?: digits)?(?: are| is|:)?|terminad[oa] en|termina en|"
    r"[uú]ltimos? (?:4|cuatro)(?: d[ií]gitos)?(?: son|:)?|\*{2,})\s*(\d{4})\b",
    re.IGNORECASE,
)
_ES_MARKERS = re.compile(
    r"[¿¡ñáéíóú]|\b(hola|gracias|cobr\w+|cargos?|reembolso|reintegro|quiero|quisiera|tarjeta|mi|por favor|para|semana|mes|ayer|"
    r"hace|cuenta|factura|suscripci[oó]n|pero|porque|necesito|pagu[eé]|buenas?)\b",
    re.IGNORECASE,
)
_EN_MARKERS = re.compile(
    r"\b(hi|hello|thanks|please|charged?|refund|want|would|like|card|my|the|was|week|month|yesterday|account|invoice|"
    r"subscription|but|because|need|paid|i)\b",
    re.IGNORECASE,
)


def detect_language(text: str) -> str:
    """"es" or "en", by counting common words. Addresses and references are ignored."""
    body = re.sub(r"\S+@\S+|\btxn\S*", " ", text)
    return "es" if len(_ES_MARKERS.findall(body)) > len(_EN_MARKERS.findall(body)) else "en"


def find_card_last4(raw_message: str) -> str | None:
    match = _CARD_LAST4.search(raw_message)
    return match.group(1) if match else None


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
        language="es" if detect_language(raw_message) == "es" else "en",
        card_last4=find_card_last4(raw_message),
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


_REASONS_ES: dict[str, str] = {
    "SYSTEM_UNAVAILABLE": "No pudimos acceder a nuestros sistemas de facturación; un especialista revisará tu caso.",
    "CUSTOMER_NOT_IDENTIFIED": "No encontramos una cuenta con los datos que nos diste.",
    "REFERENCE_NOT_ON_ACCOUNT": "No encontramos en tu cuenta la referencia de cobro que mencionaste.",
    "FRAUD_UNVERIFIED": "No pudimos verificar automáticamente el cobro que cuestionas; nuestro equipo de fraude lo va a revisar.",
    "UNAUTHORISED_TRANSACTION": "El cobro que cuestionas muestra señales de uso no autorizado; lo reembolsamos y bloqueamos la tarjeta afectada.",
    "GATEWAY_SYNC_ERROR": "Un error del procesador de pagos hizo que se te cobrara un pago informado como fallido.",
    "DUPLICATE_CHARGE": "Se te cobró dos veces la misma compra; reembolsamos el cobro duplicado.",
    "ALREADY_REFUNDED": "Este cobro ya fue reembolsado por completo; los reembolsos a tarjeta tardan de 5 a 10 días hábiles en acreditarse.",
    "CHARGE_NOT_CAPTURED": "Ese intento de pago fue rechazado y nunca se cobró; puede figurar brevemente como pendiente en tu resumen.",
    "OUT_OF_REFUND_WINDOW": "El cobro está fuera del plazo de reembolso de {days} días de tu plan.",
    "CHARGE_NOT_IDENTIFIED": "No pudimos determinar a qué cobro se refiere tu pedido.",
    "WITHIN_REFUND_WINDOW": "Tu pedido está dentro del plazo de reembolso de tu plan.",
    "CANCELLATION_REVIEW": "Un especialista en facturación tiene que revisar tu pedido de cancelación.",
    "PARTIAL_REFUND_REVIEW": "Un especialista en facturación tiene que revisar tu pedido de reembolso parcial.",
    "NO_ACTIONABLE_FINDING": "Necesitamos que un especialista revise tu pedido con más detalle.",
    "CURRENCY_UNCONFIRMED": "Nuestros cobros son en dólares estadounidenses, así que no pudimos identificar el importe que mencionaste.",
}
_MISSING_DETAILS: dict[str, dict[str, str]] = {
    "en": {
        "CHARGE_NOT_IDENTIFIED": "Please reply with the amount and date of the charge as shown on your statement.",
        "REFERENCE_NOT_ON_ACCOUNT": "Please reply with the amount and date of the charge, or the reference shown on your statement.",
        "CURRENCY_UNCONFIRMED": "Please reply with the amount in USD and the date of the charge, as shown on your statement.",
    },
    "es": {
        "CHARGE_NOT_IDENTIFIED": "Responde con el importe y la fecha del cobro tal como figuran en tu resumen.",
        "REFERENCE_NOT_ON_ACCOUNT": "Responde con el importe y la fecha del cobro, o la referencia que figura en tu resumen.",
        "CURRENCY_UNCONFIRMED": "Responde con el importe en USD y la fecha del cobro, tal como figuran en tu resumen.",
    },
}
_DEFAULT_MISSING = {
    "en": "Please reply with the email address on your account and the charge reference shown on your statement so we can continue.",
    "es": "Responde con el email de tu cuenta y la referencia del cobro que figura en tu resumen para que podamos continuar.",
}


def localized_reason(reason_code: str, english: str, language: str) -> str:
    """The customer-facing reason in the customer's language. Numbers come from the engine's own text."""
    template = _REASONS_ES.get(reason_code)
    if language != "es" or template is None:
        return english if language != "es" else "Un especialista tiene que revisar tu pedido con más detalle."
    days = re.search(r"(\d+)-day", english)
    text = template.format(days=days.group(1) if days else "")
    if "not eligible" in english:
        text += " Otro cobro que mencionaste no es elegible, por eso no está incluido en este reembolso."
    return text


def _template_draft_es(
    *,
    customer_name: str | None,
    ticket_id: str,
    action_type: str,
    executed_refunds: list[tuple[str, float]],
    reason: str,
    reason_code: str,
    human_rejected: bool,
    execution_failed: bool,
) -> CustomerDraft:
    greeting = f"Hola {customer_name.split()[0]}," if customer_name else "Hola,"
    closing = "\n\nSaludos,\nSoporte de Facturación de SentinelOps"
    ref = f"(referencia {ticket_id})"
    if action_type == "REFUND" and executed_refunds and not execution_failed:
        total = round(sum(a for _, a in executed_refunds), 2)
        lines = "\n".join(f"- ${a:,.2f} USD del cobro {tid}" for tid, a in executed_refunds)
        body = (
            f"{greeting}\n\nGracias por escribirnos sobre tu consulta de facturación {ref}. {reason}\n\n"
            f"Emitimos un reembolso por un total de ${total:,.2f} USD:\n{lines}\n\n"
            "Los reembolsos a tarjeta suelen acreditarse en 5 a 10 días hábiles, según tu banco."
        )
        return CustomerDraft(subject="Procesamos tu reembolso", body=body + closing)
    if human_rejected:
        body = (
            f"{greeting}\n\nGracias por tu paciencia {ref}. Nuestros especialistas revisaron tu pedido y por ahora no "
            "pudieron aprobar un reembolso. Una persona de nuestro equipo te va a contactar dentro de un día hábil "
            "para explicarte el resultado y los próximos pasos."
        )
        return CustomerDraft(subject="Novedades sobre tu pedido de facturación", body=body + closing)
    if execution_failed:
        body = (
            f"{greeting}\n\nGracias por escribirnos {ref}. Aprobamos tu pedido, pero tuvimos un problema temporal al "
            "procesarlo. Un especialista en facturación se está ocupando personalmente y te va a confirmar cuando esté listo."
        )
        return CustomerDraft(subject="Estamos procesando tu pedido", body=body + closing)
    if action_type == "DENY":
        body = (
            f"{greeting}\n\nGracias por escribirnos {ref}. Revisamos tu cuenta con atención. {reason} "
            "Si crees que es un error, responde este email y tu ejecutivo de cuenta lo va a revisar."
        )
        return CustomerDraft(subject="Sobre tu pedido de reembolso", body=body + closing)
    if action_type == "REQUEST_INFO":
        needed = _MISSING_DETAILS["es"].get(reason_code, _DEFAULT_MISSING["es"])
        body = f"{greeting}\n\nGracias por escribirnos {ref}. {reason} {needed}"
        return CustomerDraft(subject="Necesitamos algunos datos más", body=body + closing)
    body = (
        f"{greeting}\n\nGracias por escribirnos {ref}. {reason} "
        "Un especialista en facturación va a revisar tu caso y te va a responder dentro de un día hábil."
    )
    return CustomerDraft(subject="Estamos revisando tu caso", body=body + closing)


def template_draft(
    *,
    customer_name: str | None,
    ticket_id: str,
    action_type: str,
    executed_refunds: list[tuple[str, float]],
    customer_facing_reason: str,
    human_rejected: bool,
    execution_failed: bool,
    language: str = "en",
    reason_code: str = "",
) -> CustomerDraft:
    if language == "es":
        return _template_draft_es(
            customer_name=customer_name,
            ticket_id=ticket_id,
            action_type=action_type,
            executed_refunds=executed_refunds,
            reason=localized_reason(reason_code, customer_facing_reason, "es"),
            reason_code=reason_code,
            human_rejected=human_rejected,
            execution_failed=execution_failed,
        )
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
        needed = _MISSING_DETAILS["en"].get(reason_code, _DEFAULT_MISSING["en"])
        body = f"{greeting}\n\nThank you for contacting us {ref}. {customer_facing_reason} {needed}"
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
    refund_executed: bool = True,
) -> JudgeVerdict:
    report = grounding_check(
        draft,
        allowed_amounts=allowed_amounts,
        allowed_transaction_ids=allowed_transaction_ids,
        required_amounts=required_amounts,
        refund_executed=refund_executed,
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
