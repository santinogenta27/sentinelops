"""Auto-Pilot (zero-touch) policy for the HumanGate.

Auto-Pilot never bypasses the gate: the graph still pauses at ``interrupt()``; the runtime then
resumes it with an ``auto-pilot`` decision *only* for cases the company policy lets a machine
authorise. Anything risky keeps waiting for a human supervisor:

* prompt-injection / manipulation signals          → human
* sender is not the account owner (POL-ID-001)      → human
* risk level HIGH (fraud claims, chargeback abuse…) → human
* payout above ``AUTOPILOT_MAX_USD``                 → human
* anything that is not a plain REFUND              → human
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

AUTOPILOT_REVIEWER = "auto-pilot"


@dataclass(frozen=True, slots=True)
class AutopilotVerdict:
    eligible: bool
    reasons: list[str] = field(default_factory=list)  # why it was NOT eligible (empty when eligible)
    amount: float = 0.0


def autopilot_verdict(values: Mapping[str, Any], *, enabled: bool, max_usd: float) -> AutopilotVerdict:
    action = values.get("resolution_action") or {}
    risk = values.get("risk") or {}
    amount = round(sum(float(r.get("amount", 0)) for r in action.get("refunds") or []), 2)
    reasons: list[str] = []
    if not enabled:
        reasons.append("auto-pilot is OFF")
    if values.get("injection_signals"):
        reasons.append("prompt-injection signals detected")
    flags = [str(f) for f in risk.get("flags", [])]
    if "sender_not_account_owner" in flags:
        reasons.append("sender is not the account owner (POL-ID-001)")
    if risk.get("level") == "HIGH":
        reasons.append(f"risk level HIGH ({', '.join(flags) or 'no flags'})")
    if action.get("action_type") != "REFUND":
        reasons.append(f"action {action.get('action_type')} is not a plain refund")
    if amount > max_usd:
        reasons.append(f"payout ${amount:,.2f} exceeds the auto-pilot ceiling ${max_usd:,.2f}")
    return AutopilotVerdict(eligible=not reasons, reasons=reasons, amount=amount)
