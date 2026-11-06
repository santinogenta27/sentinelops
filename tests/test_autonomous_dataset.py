"""Messy, previously unseen emails through the email channel: the gate for free-form disputes.

Expected outcomes come from the policy handbook and the ledger, never from what the agent does.
Unlike the held-out set, this dataset is meant to be extended whenever a new real-world failure shows up.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from app.db.database import iso
from app.integrations.email_inbound import build_inbound_email
from app.models.schemas import HumanDecision
from app.services.inbound_email import InboundEmailService
from app.services.runtime import SentinelRuntime

CASES: list[dict[str, Any]] = json.loads((Path(__file__).parent / "datasets" / "autonomous_tickets.json").read_text(encoding="utf-8"))


async def _setup(rt: SentinelRuntime, charges: list[dict[str, Any]]) -> None:
    async with rt.db.connect() as conn:
        for c in charges:
            await conn.execute(
                "INSERT INTO transactions (id, customer_id, amount, currency, status, gateway_status, created_at, "
                "merchant_descriptor, card_last4, ip_country, refunded_amount) "
                "VALUES (?, ?, ?, 'USD', 'SUCCEEDED', 'CAPTURED', ?, ?, '4242', 'GB', 0)",
                (c["id"], c["customer_id"], c["amount"], iso(datetime.now(UTC) - timedelta(days=c["days_ago"])), c["descriptor"]),
            )
        await conn.commit()


def test_dataset_is_well_formed() -> None:
    assert len(CASES) >= 20 and len({c["id"] for c in CASES}) == len(CASES)
    for c in CASES:
        exp = c["expected"]
        assert exp["requires_approval"] == ("decision" in c), f"{c['id']}: gated cases need a decision and only those"
        assert c["rationale"], f"{c['id']}: every expectation is justified"


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
async def test_case(runtime: SentinelRuntime, case: dict[str, Any]) -> None:
    await _setup(runtime, case.get("setup", []))
    exp = case["expected"]
    email = build_inbound_email(raw_from=case["sender"], subject=case["subject"], body=case["message"],
                                message_id=f"<{case['id']}@mail>", source="simulator")
    result = await InboundEmailService(runtime).ingest(email, wait=True)
    view = await runtime.get_view(result.ticket_id)

    assert view.requires_human_approval is exp["requires_approval"]
    if exp["requires_approval"]:
        assert await runtime.db.list_refunds() == [], "no money moves before the human decision"
        approved = case["decision"] == "approve"
        await runtime.resume(result.ticket_id, HumanDecision(approved=approved, reviewer="eval.supervisor",
                                                             reason=None if approved else "eval: rejected"))
        view = await runtime.get_view(result.ticket_id)

    action = view.resolution_action or {}
    assert action.get("action_type") == exp["action_type"]
    assert view.execution_status == exp["final_status"]
    assert round(sum(float(r["amount"]) for r in await runtime.db.list_refunds()), 2) == exp["payout"]
    assert view.customer_language == exp["language"]
    assert (view.judge_verdict or {}).get("passed") is True, "the reply passes the grounding audit"
    assert bool(view.injection_signals) is bool(case.get("adversarial"))
    for text in case.get("forbidden_text", []):
        assert text not in view.customer_response_draft
    if "queue" in case:
        queued = await runtime.db.get_case(result.ticket_id)
        assert queued is not None and queued["reason_code"] == case["queue"]
