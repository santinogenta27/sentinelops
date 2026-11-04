"""Deterministic financial decision engine.

Design principle: the LLM *understands* language; this module *decides money*.
Payout amounts are always derived from verified ledger records, never from the
customer's text or from free-form model output — which makes the system immune to
hallucinated amounts and to prompt-injected payout instructions by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from app.models.schemas import (
    ClaimCategory,
    CustomerRecord,
    Finding,
    RefundInstruction,
    ResolutionAction,
    RiskAssessment,
    RiskLevel,
    SecondaryAction,
    TransactionRecord,
)

DUPLICATE_WINDOW = timedelta(minutes=10)
REFUND_WINDOW_DAYS: dict[str, int] = {"FREE": 14, "PRO": 30, "ENTERPRISE": 60}
GATEWAY_FAILURE_STATES = frozenset({"FAILED", "TIMEOUT", "VOIDED"})
CAPTURED_LEDGER_STATES = frozenset({"SUCCEEDED", "PARTIALLY_REFUNDED"})
NOT_CAPTURED_LEDGER_STATES = frozenset({"FAILED", "PENDING"})
# POL-FRD-001 "abnormal amounts": a charge far above the customer's own history (needs some history).
ABNORMAL_AMOUNT_MULTIPLIER = 5.0
ABNORMAL_MIN_HISTORY = 2
MAX_LOOKBACK_DAYS = 365


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------
def detect_duplicates(txns: list[TransactionRecord]) -> list[Finding]:
    findings: list[Finding] = []
    groups: dict[tuple[float, str], list[TransactionRecord]] = {}
    for t in txns:
        if t.status in CAPTURED_LEDGER_STATES and t.gateway_status == "CAPTURED":
            groups.setdefault((t.amount, t.merchant_descriptor), []).append(t)
    for (amount, descriptor), group in groups.items():
        group.sort(key=lambda t: t.created_at)
        for prev, cur in zip(group, group[1:], strict=False):
            if cur.created_at - prev.created_at <= DUPLICATE_WINDOW and cur.refundable_amount > 0:
                findings.append(
                    Finding(
                        kind="DUPLICATE_CHARGE",
                        transaction_ids=[prev.id, cur.id],
                        detail=(
                            f"{cur.id} duplicates {prev.id}: same amount {amount:.2f} and descriptor "
                            f"'{descriptor}' {int((cur.created_at - prev.created_at).total_seconds())}s apart"
                        ),
                    )
                )
    return findings


def detect_gateway_sync_errors(txns: list[TransactionRecord]) -> list[Finding]:
    return [
        Finding(
            kind="GATEWAY_SYNC_ERROR",
            transaction_ids=[t.id],
            detail=f"ledger shows {t.id} captured ({t.amount:.2f}) but gateway reported {t.gateway_status}",
        )
        for t in txns
        if t.status in CAPTURED_LEDGER_STATES and t.gateway_status in GATEWAY_FAILURE_STATES and t.refundable_amount > 0
    ]


def detect_fraud_signals(customer: CustomerRecord, txns: list[TransactionRecord]) -> list[Finding]:
    findings: list[Finding] = []
    ordered = sorted(txns, key=lambda t: t.created_at)
    seen_cards: set[str] = set()
    for i, t in enumerate(ordered):
        signals: list[str] = []
        if t.ip_country != customer.country:
            signals.append(f"IP country {t.ip_country} differs from account country {customer.country}")
        if i > 0 and t.card_last4 not in seen_cards:
            signals.append(f"card ending {t.card_last4} not previously used on this account")
        prior = [p.amount for p in ordered[:i]]
        if len(prior) >= ABNORMAL_MIN_HISTORY and t.amount > ABNORMAL_AMOUNT_MULTIPLIER * max(prior):
            signals.append(f"amount {t.amount:.2f} is {t.amount / max(prior):.0f}x the largest previous charge ({max(prior):.2f})")
        seen_cards.add(t.card_last4)
        if signals and t.refundable_amount > 0:
            findings.append(Finding(kind="FRAUD_SIGNAL", transaction_ids=[t.id], detail="; ".join(signals)))
    return findings


def _claimed(dispute_amount: float | None, claimed_amounts: list[float] | None) -> list[float]:
    out: list[float] = []
    for a in ([dispute_amount] if dispute_amount is not None else []) + list(claimed_amounts or []):
        if all(abs(a - b) >= 0.005 for b in out):
            out.append(round(a, 2))
    return out


def select_targets(
    txns: list[TransactionRecord],
    *,
    referenced_ids: list[str],
    dispute_amount: float | None,
    claimed_amounts: list[float] | None = None,
) -> list[TransactionRecord]:
    """Which transactions is the customer talking about? Explicit ids > amount match > the only charge.

    Never guesses: when the customer cites ids or amounts that are not on the ledger, or cites
    nothing while several charges exist, no target is returned (the planner asks for clarification
    instead of moving money on a charge the customer did not dispute).
    """
    if referenced_ids:
        return [t for t in txns if t.id in referenced_ids]
    amounts = _claimed(dispute_amount, claimed_amounts)
    if amounts:
        targets: list[TransactionRecord] = []
        for a in amounts:
            matched = [t for t in txns if abs(t.amount - a) < 0.005]
            if matched:
                best = max(matched, key=lambda t: t.created_at)
                if best not in targets:
                    targets.append(best)
        return targets
    return list(txns) if len(txns) == 1 else []


@dataclass(slots=True)
class Investigation:
    customer: CustomerRecord | None
    transactions: list[TransactionRecord]
    targets: list[TransactionRecord]
    findings: list[Finding] = field(default_factory=list)
    missing_references: list[str] = field(default_factory=list)


def investigate(
    *,
    customer: CustomerRecord | None,
    txns: list[TransactionRecord],
    category: ClaimCategory,
    referenced_ids: list[str],
    dispute_amount: float | None,
    now: datetime,
    claimed_amounts: list[float] | None = None,
) -> Investigation:
    if customer is None:
        return Investigation(
            customer=None,
            transactions=[],
            targets=[],
            findings=[Finding(kind="CUSTOMER_NOT_FOUND", detail="no account matches the supplied identifiers")],
        )
    present = {t.id for t in txns}
    inv = Investigation(
        customer=customer,
        transactions=txns,
        targets=select_targets(
            txns, referenced_ids=referenced_ids, dispute_amount=dispute_amount, claimed_amounts=claimed_amounts
        ),
        missing_references=[r for r in referenced_ids if r not in present],
    )
    if not txns:
        inv.findings.append(Finding(kind="NO_MATCHING_TRANSACTION", detail="no transactions in the lookback window"))
        return inv
    if inv.missing_references:
        inv.findings.append(
            Finding(kind="REFERENCE_NOT_ON_ACCOUNT", detail=f"cited charges {inv.missing_references} are not on this account")
        )
    elif not inv.targets:
        if _claimed(dispute_amount, claimed_amounts):
            inv.findings.append(Finding(kind="AMOUNT_NOT_ON_LEDGER", detail="no charge matches the amounts the customer cited"))
        else:
            inv.findings.append(Finding(kind="TARGET_AMBIGUOUS", detail=f"{len(txns)} charges on file and none was specified"))
    inv.findings.extend(detect_duplicates(txns))
    inv.findings.extend(detect_gateway_sync_errors(txns))
    if category == "FRAUD_CLAIM":
        inv.findings.extend(detect_fraud_signals(customer, txns))
    window = REFUND_WINDOW_DAYS[customer.tier]
    for t in inv.targets:
        if t.status in NOT_CAPTURED_LEDGER_STATES:
            inv.findings.append(
                Finding(kind="NOT_CAPTURED", transaction_ids=[t.id], detail=f"{t.id} was never captured (status {t.status})")
            )
        elif t.status == "REFUNDED" or t.refundable_amount <= 0:
            inv.findings.append(
                Finding(kind="ALREADY_REFUNDED", transaction_ids=[t.id], detail=f"{t.id} already refunded ({t.refunded_amount:.2f})")
            )
        elif now - t.created_at > timedelta(days=window):
            age = (now - t.created_at).days
            inv.findings.append(
                Finding(
                    kind="OUT_OF_REFUND_WINDOW",
                    transaction_ids=[t.id],
                    detail=f"{t.id} is {age} days old; {customer.tier} window is {window} days",
                )
            )
        else:
            inv.findings.append(
                Finding(kind="ELIGIBLE_REFUND", transaction_ids=[t.id], detail=f"{t.id} within {window}-day {customer.tier} window")
            )
    if dispute_amount is not None:
        known = {round(t.amount, 2) for t in txns}
        if all(abs(dispute_amount - k) >= 0.005 for k in known):
            inv.findings.append(
                Finding(
                    kind="AMOUNT_MISMATCH",
                    detail=f"claimed amount {dispute_amount:.2f} does not match any ledger amount",
                )
            )
    return inv


def inflated_amount_requested(txns: list[TransactionRecord], claimed_amounts: list[float]) -> bool:
    """POL-SEC-001: the message cites a real ledger charge but asks for a larger, unsupported amount
    (e.g. "refund $490 — the $49 charge plus compensation"). Treated as a manipulation attempt."""
    ledger = {round(t.amount, 2) for t in txns}
    on_ledger = [a for a in claimed_amounts if any(abs(a - b) < 0.005 for b in ledger)]
    unsupported = [a for a in claimed_amounts if all(abs(a - b) >= 0.005 for b in ledger)]
    return bool(on_ledger) and any(u > max(on_ledger) for u in unsupported)


# ---------------------------------------------------------------------------
# Risk
# ---------------------------------------------------------------------------
def assess_risk(
    *,
    customer: CustomerRecord | None,
    findings: list[Finding],
    category: ClaimCategory,
    injection_signals: list[str],
    payout_total: float,
    hitl_threshold: float,
    now: datetime,
    sender_mismatch: bool = False,
) -> RiskAssessment:
    score = 0.0
    flags: list[str] = []
    if sender_mismatch:
        # Request came from an address that does not own the account (POL-ID-001): identity unverified.
        score += 0.6
        flags.append("sender_not_account_owner")
    kinds = {f.kind for f in findings}
    if category == "FRAUD_CLAIM":
        score += 0.6
        flags.append("fraud_claim")
    if "FRAUD_SIGNAL" in kinds:
        score += 0.2
        flags.append("fraud_signals_on_ledger")
    if injection_signals:
        score += 0.6
        flags.append("prompt_injection:" + ",".join(injection_signals))
    if customer is not None:
        # POL-FRD-002: either condition alone makes the account HIGH risk, so any payout needs a human.
        if customer.chargeback_count >= 3:
            score += 0.6
            flags.append(f"chargeback_history:{customer.chargeback_count}")
        if now - customer.account_created_at < timedelta(days=7):
            score += 0.6
            flags.append("new_account_lt_7d")
        if customer.account_status != "ACTIVE":
            score += 0.4
            flags.append(f"account_{customer.account_status.lower()}")
    if "AMOUNT_MISMATCH" in kinds:
        score += 0.15
        flags.append("claimed_amount_mismatch")
    if payout_total > hitl_threshold:
        score += 0.2
        flags.append(f"payout_above_{hitl_threshold:.0f}")
    score = round(min(score, 1.0), 2)
    level: RiskLevel = "HIGH" if score >= 0.6 else "MEDIUM" if score >= 0.3 else "LOW"
    return RiskAssessment(score=score, level=level, flags=flags)


def requires_human_approval(
    action: ResolutionAction, risk: RiskAssessment, *, injection_detected: bool, hitl_threshold: float
) -> bool:
    """SPEC gate: payout > threshold OR (payout on high risk) OR manipulation attempt detected."""
    if injection_detected:
        return True
    if action.action_type != "REFUND":
        return False
    return action.total_amount > hitl_threshold or risk.level == "HIGH"


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class PlanResult:
    """Either a concrete action, or a request to re-run DBInspector with a wider lookback."""

    action: ResolutionAction | None
    requery_lookback_days: int | None = None


def _refund_action(
    refunds: dict[str, float], reason_code: str, policies: list[str], secondary: list[SecondaryAction], reason: str
) -> ResolutionAction:
    return ResolutionAction(
        action_type="REFUND",
        refunds=[RefundInstruction(transaction_id=k, amount=round(v, 2)) for k, v in refunds.items()],
        reason_code=reason_code,
        policy_refs=policies,
        secondary_actions=secondary,
        customer_facing_reason=reason,
    )


def plan_resolution(
    *,
    inv: Investigation,
    category: ClaimCategory,
    lookback_days: int,
    tool_failure: bool,
    injection_detected: bool,
    refund_requested: bool = False,
) -> PlanResult:
    """``refund_requested``: the customer explicitly asked for money back. Without it, only the
    policy-mandated refunds (duplicates, gateway errors, verified fraud) are made — a plain
    in-window refund is never issued on, e.g., a cancellation request or a billing question."""
    if tool_failure:
        return PlanResult(
            ResolutionAction(
                action_type="ESCALATE",
                reason_code="SYSTEM_UNAVAILABLE",
                policy_refs=["POL-ESC-001"],
                secondary_actions=["CRM_NOTE"],
                customer_facing_reason="Our billing systems could not be reached; a specialist will review the case.",
            )
        )
    if inv.customer is None:
        return PlanResult(
            ResolutionAction(
                action_type="REQUEST_INFO",
                reason_code="CUSTOMER_NOT_IDENTIFIED",
                policy_refs=["POL-ID-001"],
                customer_facing_reason="We could not locate an account for the details provided; we need the account email and the charge reference.",
            )
        )
    if inv.missing_references and lookback_days < MAX_LOOKBACK_DAYS:
        # Planner -> DBInspector loop: the customer cited charges we have not loaded yet.
        return PlanResult(action=None, requery_lookback_days=MAX_LOOKBACK_DAYS)
    if inv.missing_references:
        # POL-ID-001: the cited charge is not on this account (does not exist / belongs to someone else).
        return PlanResult(
            ResolutionAction(
                action_type="REQUEST_INFO",
                reason_code="REFERENCE_NOT_ON_ACCOUNT",
                policy_refs=["POL-ID-001"],
                secondary_actions=["CRM_NOTE"],
                customer_facing_reason="We could not find the charge reference you mentioned on your account.",
            )
        )

    by_id = {t.id: t for t in inv.transactions}
    kinds = {f.kind for f in inv.findings}
    secondary: list[SecondaryAction] = ["CRM_NOTE"]
    risky_account = inv.customer.chargeback_count >= 3

    if category == "FRAUD_CLAIM":
        disputed = {t.id for t in inv.targets}
        # Refund only charges the customer disputes AND that carry fraud signals — never another charge.
        flagged = [tid for f in inv.findings if f.kind == "FRAUD_SIGNAL" for tid in f.transaction_ids if tid in disputed]
        if not flagged:
            return PlanResult(
                ResolutionAction(
                    action_type="ESCALATE",
                    reason_code="FRAUD_UNVERIFIED",
                    policy_refs=["POL-FRD-001", "POL-HITL-001"],
                    secondary_actions=["FLAG_ACCOUNT", "CRM_NOTE"],
                    customer_facing_reason="We could not automatically verify the disputed charge; our fraud team will review it.",
                )
            )
        fraud_refunds = {tid: by_id[tid].refundable_amount for tid in flagged}
        return PlanResult(
            _refund_action(
                fraud_refunds,
                "UNAUTHORISED_TRANSACTION",
                ["POL-FRD-001", "POL-HITL-001"],
                ["FREEZE_CARD", "FLAG_ACCOUNT", "CRM_NOTE"],
                "The disputed charge shows signs of unauthorised use; it will be refunded and the affected card blocked.",
            )
        )

    refunds: dict[str, float] = {}
    policies: list[str] = []
    reason_code = ""
    reason = ""
    if "GATEWAY_SYNC_ERROR" in kinds:
        for f in inv.findings:
            if f.kind == "GATEWAY_SYNC_ERROR":
                for tid in f.transaction_ids:
                    refunds[tid] = by_id[tid].refundable_amount
        policies.append("POL-GW-001")
        secondary.append("RECONCILE_GATEWAY")
        reason_code = "GATEWAY_SYNC_ERROR"
        reason = "A payment-processor error caused you to be charged for a payment that was reported as failed."
    if "DUPLICATE_CHARGE" in kinds:
        # POL-DUP-001: every later charge in a duplicate chain is refunded (A, A+2min, A+4min -> refund
        # the 2nd and 3rd); only the original charge is retained.
        for f in inv.findings:
            if f.kind == "DUPLICATE_CHARGE":
                dup = f.transaction_ids[-1]
                if dup not in refunds:
                    refunds[dup] = by_id[dup].refundable_amount
        policies.append("POL-DUP-001")
        reason_code = reason_code or "DUPLICATE_CHARGE"
        reason = reason or "You were charged twice for the same purchase; the duplicate charge will be refunded."

    if not refunds:
        if category in ("REFUND_REQUEST", "DUPLICATE_CHARGE"):
            if "ALREADY_REFUNDED" in kinds:
                return PlanResult(
                    ResolutionAction(
                        action_type="DENY",
                        reason_code="ALREADY_REFUNDED",
                        policy_refs=["POL-REF-003"],
                        secondary_actions=["CRM_NOTE"],
                        customer_facing_reason="This charge has already been refunded in full; card refunds take 5-10 business days to settle.",
                    )
                )
            if "NOT_CAPTURED" in kinds:
                return PlanResult(
                    ResolutionAction(
                        action_type="DENY",
                        reason_code="CHARGE_NOT_CAPTURED",
                        policy_refs=["POL-DUP-001"],
                        secondary_actions=["CRM_NOTE"],
                        customer_facing_reason=(
                            "That payment attempt was declined and never charged, so you were only billed once; "
                            "it may show briefly as pending on your statement."
                        ),
                    )
                )
            if "OUT_OF_REFUND_WINDOW" in kinds:
                window = REFUND_WINDOW_DAYS[inv.customer.tier]
                return PlanResult(
                    ResolutionAction(
                        action_type="DENY",
                        reason_code="OUT_OF_REFUND_WINDOW",
                        policy_refs=["POL-REF-002", "POL-REF-001"],
                        secondary_actions=["CRM_NOTE"],
                        customer_facing_reason=f"The charge is outside the {window}-day refund window for your plan.",
                    )
                )
            if kinds & {"AMOUNT_NOT_ON_LEDGER", "TARGET_AMBIGUOUS"}:
                return PlanResult(
                    ResolutionAction(
                        action_type="REQUEST_INFO",
                        reason_code="CHARGE_NOT_IDENTIFIED",
                        policy_refs=["POL-ID-001"],
                        secondary_actions=["CRM_NOTE"],
                        customer_facing_reason="We could not tell which charge your request refers to.",
                    )
                )
            eligible = [tid for f in inv.findings if f.kind == "ELIGIBLE_REFUND" for tid in f.transaction_ids]
            if eligible and refund_requested:
                refunds = {tid: by_id[tid].refundable_amount for tid in eligible}
                policies.append("POL-REF-001")
                reason_code = "WITHIN_REFUND_WINDOW"
                reason = "Your request falls within the refund window for your plan."
        if not refunds:
            return PlanResult(
                ResolutionAction(
                    action_type="ESCALATE",
                    reason_code="NO_ACTIONABLE_FINDING",
                    policy_refs=["POL-COM-001"],
                    secondary_actions=["CRM_NOTE"],
                    customer_facing_reason="We need a specialist to review your request in more detail.",
                )
            )

    if risky_account or injection_detected:
        secondary.append("FLAG_ACCOUNT")
        policies.append("POL-FRD-002" if risky_account else "POL-SEC-001")
    return PlanResult(_refund_action(refunds, reason_code, policies, secondary, reason))
