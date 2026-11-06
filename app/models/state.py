"""LangGraph state schema (``AgentState``).

Mirrors SPEC §4.1 and adds the operational fields needed for risk gating,
self-correction, prompt-injection defence and observability. Append-only
channels (``logs``, ``tool_calls``) use an ``operator.add`` reducer so every
node only returns its *delta*.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict


class AgentState(TypedDict):
    # --- SPEC §4.1 core fields -------------------------------------------
    ticket_id: str
    raw_message: str
    customer_email: Optional[str]
    dispute_amount: Optional[float]
    claim_category: Optional[str]  # DUPLICATE_CHARGE | FRAUD_CLAIM | REFUND_REQUEST | GENERAL
    customer_tier: Optional[str]  # ENTERPRISE | PRO | FREE
    db_records: List[Dict[str, Any]]
    policy_clauses: List[str]
    investigation_summary: str
    requires_human_approval: bool
    human_approved: Optional[bool]
    rejection_reason: Optional[str]
    tool_error_count: int
    resolution_action: Optional[Dict[str, Any]]
    customer_response_draft: str
    judge_verdict: Optional[Dict[str, Any]]  # {passed, faithfulness_score, feedback, issues}
    execution_status: str
    logs: Annotated[List[str], operator.add]

    # --- Operational extensions ------------------------------------------
    transaction_refs: List[str]
    customer_profile: Optional[Dict[str, Any]]
    findings: List[Dict[str, Any]]
    risk: Optional[Dict[str, Any]]
    last_tool_error: Optional[str]
    tool_query_args: Optional[Dict[str, Any]]
    requery_requested: bool
    tool_calls: Annotated[List[Dict[str, Any]], operator.add]
    injection_signals: List[str]
    triage_signals: Dict[str, Any]  # TriageSignals: dates, card, product, intents verified against the message
    customer_language: str  # "en" | "es" | "other"
    complaint_summary: str  # what the customer asked, from triage (investigation_summary becomes internal)
    reviewer: Optional[str]
    execution_receipt: Optional[Dict[str, Any]]
    draft_attempts: int
    email_dispatched: bool

    # --- Channel context (inbound email) -----------------------------------
    channel: str  # "web" | "email"
    inbound_source: Optional[str]  # "webhook" | "imap" | "simulator"
    sender_email: Optional[str]  # address the request actually came from
    email_subject: Optional[str]
    email_message_id: Optional[str]
    sender_alias_of: Optional[str]  # account email this sender is a verified alias of (operator-linked)
    identity_unverified: bool  # identity rests on the message text or a failed sender check (POL-ID-001)
    parent_ticket_id: Optional[str]  # the case this email answers, when it is a clarification reply
    clarification_round: int

    # --- Stripe live mirror (test mode) ------------------------------------
    stripe_mirror: List[Dict[str, Any]]


def initial_state(
    ticket_id: str,
    raw_message: str,
    customer_email: Optional[str] = None,
    *,
    channel: str = "web",
    inbound_source: Optional[str] = None,
    sender_email: Optional[str] = None,
    email_subject: Optional[str] = None,
    email_message_id: Optional[str] = None,
    sender_alias_of: Optional[str] = None,
    identity_unverified: bool = False,
    parent_ticket_id: Optional[str] = None,
    clarification_round: int = 0,
) -> AgentState:
    return AgentState(
        sender_alias_of=sender_alias_of,
        identity_unverified=identity_unverified,
        parent_ticket_id=parent_ticket_id,
        clarification_round=clarification_round,
        complaint_summary="",
        stripe_mirror=[],
        channel=channel,
        inbound_source=inbound_source,
        sender_email=sender_email,
        email_subject=email_subject,
        email_message_id=email_message_id,
        ticket_id=ticket_id,
        raw_message=raw_message,
        customer_email=customer_email,
        dispute_amount=None,
        claim_category=None,
        customer_tier=None,
        db_records=[],
        policy_clauses=[],
        investigation_summary="",
        requires_human_approval=False,
        human_approved=None,
        rejection_reason=None,
        tool_error_count=0,
        resolution_action=None,
        customer_response_draft="",
        judge_verdict=None,
        execution_status="TRIAGING",
        logs=[],
        transaction_refs=[],
        customer_profile=None,
        findings=[],
        risk=None,
        last_tool_error=None,
        tool_query_args=None,
        requery_requested=False,
        tool_calls=[],
        injection_signals=[],
        triage_signals={},
        customer_language="en",
        reviewer=None,
        execution_receipt=None,
        draft_attempts=0,
        email_dispatched=False,
    )
