"""Strict Pydantic v2 domain models.

Every boundary in the system (LLM output, tool arguments, DB rows, API payloads,
human decisions) is validated through one of these models. LangGraph state stores
their JSON-serialisable ``model_dump(mode="json")`` form so the SQLite checkpointer
can persist it safely; nodes always re-hydrate through the models below.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

# ---------------------------------------------------------------------------
# Primitive types
# ---------------------------------------------------------------------------
EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,24}$")
TXN_ID_PATTERN = r"^txn_[A-Za-z0-9]{4,32}$"
TICKET_ID_PATTERN = r"^TCK-[A-Z0-9]{8}$"

TransactionId = Annotated[str, StringConstraints(pattern=TXN_ID_PATTERN)]
TicketId = Annotated[str, StringConstraints(pattern=TICKET_ID_PATTERN)]
CurrencyCode = Annotated[str, StringConstraints(pattern=r"^[A-Z]{3}$")]
Money = Annotated[float, Field(ge=0, le=1_000_000)]

ClaimCategory = Literal["DUPLICATE_CHARGE", "FRAUD_CLAIM", "REFUND_REQUEST", "GENERAL"]
CustomerTier = Literal["ENTERPRISE", "PRO", "FREE"]
RiskLevel = Literal["LOW", "MEDIUM", "HIGH"]
ExecutionStatus = Literal[
    "TRIAGING",
    "INVESTIGATING",
    "PLANNING",
    "AWAITING_APPROVAL",
    "EXECUTING",
    "DRAFTING",
    "AUDITING",
    "EXECUTED",
    "REJECTED",
    "ESCALATED",
    "FAILED",
]
TERMINAL_STATUSES: frozenset[str] = frozenset({"EXECUTED", "REJECTED", "ESCALATED", "FAILED"})

ActionType = Literal["REFUND", "DENY", "ESCALATE", "REQUEST_INFO"]
SecondaryAction = Literal["FREEZE_CARD", "FLAG_ACCOUNT", "RECONCILE_GATEWAY", "CRM_NOTE"]
FindingKind = Literal[
    "DUPLICATE_CHARGE",
    "GATEWAY_SYNC_ERROR",
    "FRAUD_SIGNAL",
    "ALREADY_REFUNDED",
    "OUT_OF_REFUND_WINDOW",
    "ELIGIBLE_REFUND",
    "NO_MATCHING_TRANSACTION",
    "AMOUNT_MISMATCH",
    "CUSTOMER_NOT_FOUND",
]


def normalise_email(value: str | None) -> str | None:
    if value is None:
        return None
    candidate = value.strip().lower()
    if not candidate:
        return None
    if not EMAIL_RE.fullmatch(candidate):
        raise ValueError(f"invalid email address: {value!r}")
    return candidate


class StrictModel(BaseModel):
    """Base for all domain models: no extra keys, assignment validation, frozen-ish."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, str_strip_whitespace=True)


# ---------------------------------------------------------------------------
# LLM structured outputs
# ---------------------------------------------------------------------------
class TriageExtraction(StrictModel):
    """Entities the TriageExtractor pulls out of an unstructured complaint."""

    customer_email: str | None = Field(default=None, description="Customer email if explicitly present")
    dispute_amount: Money | None = Field(default=None, description="Disputed amount in the message, if any")
    currency: CurrencyCode = "USD"
    claim_category: ClaimCategory = "GENERAL"
    transaction_refs: list[Annotated[str, StringConstraints(max_length=64)]] = Field(
        default_factory=list,
        max_length=10,
        description="Transaction references literally mentioned by the customer (unvalidated)",
    )
    summary: Annotated[str, StringConstraints(min_length=1, max_length=600)] = Field(
        description="One-paragraph neutral summary of the complaint"
    )

    @field_validator("customer_email", mode="before")
    @classmethod
    def _email(cls, v: object) -> str | None:
        if v is None or v == "":
            return None
        if not isinstance(v, str):
            raise ValueError("customer_email must be a string")
        return normalise_email(v)

    @field_validator("dispute_amount", mode="after")
    @classmethod
    def _round(cls, v: float | None) -> float | None:
        return None if v is None else round(v, 2)


class JudgeVerdict(StrictModel):
    passed: bool
    faithfulness_score: Annotated[float, Field(ge=0.0, le=1.0)]
    feedback: Annotated[str, StringConstraints(max_length=1500)]
    issues: list[Annotated[str, StringConstraints(max_length=300)]] = Field(default_factory=list, max_length=20)


class InvestigationNarrative(StrictModel):
    summary: Annotated[str, StringConstraints(min_length=1, max_length=1500)]


class CustomerDraft(StrictModel):
    subject: Annotated[str, StringConstraints(min_length=1, max_length=160)]
    body: Annotated[str, StringConstraints(min_length=1, max_length=4000)]


# ---------------------------------------------------------------------------
# Tool arguments & records
# ---------------------------------------------------------------------------
class TransactionQuery(StrictModel):
    """Arguments of the ``query_transactions`` tool. Invalid args trigger self-correction."""

    customer_email: str | None = None
    transaction_ids: list[TransactionId] = Field(default_factory=list, max_length=10)
    lookback_days: Annotated[int, Field(ge=1, le=365)] = 90

    @field_validator("customer_email", mode="before")
    @classmethod
    def _email(cls, v: object) -> str | None:
        if v is None or v == "":
            return None
        if not isinstance(v, str):
            raise ValueError("customer_email must be a string")
        return normalise_email(v)

    @model_validator(mode="after")
    def _at_least_one_filter(self) -> TransactionQuery:
        if self.customer_email is None and not self.transaction_ids:
            raise ValueError("query requires customer_email or at least one transaction_id")
        return self


class CustomerRecord(StrictModel):
    id: Annotated[str, StringConstraints(pattern=r"^cus_[A-Za-z0-9]{4,32}$")]
    email: str
    name: Annotated[str, StringConstraints(min_length=1, max_length=120)]
    tier: CustomerTier
    country: Annotated[str, StringConstraints(pattern=r"^[A-Z]{2}$")]
    account_created_at: datetime
    chargeback_count: Annotated[int, Field(ge=0)]
    account_status: Literal["ACTIVE", "SUSPENDED", "CLOSED"]


class TransactionRecord(StrictModel):
    id: TransactionId
    customer_id: str
    amount: Money
    currency: CurrencyCode
    status: Literal["SUCCEEDED", "REFUNDED", "PARTIALLY_REFUNDED", "FAILED", "PENDING"]
    gateway_status: Literal["CAPTURED", "FAILED", "TIMEOUT", "VOIDED", "PENDING"]
    created_at: datetime
    merchant_descriptor: Annotated[str, StringConstraints(max_length=120)]
    card_last4: Annotated[str, StringConstraints(pattern=r"^\d{4}$")]
    ip_country: Annotated[str, StringConstraints(pattern=r"^[A-Z]{2}$")]
    refunded_amount: Money = 0.0

    @property
    def refundable_amount(self) -> float:
        return round(max(self.amount - self.refunded_amount, 0.0), 2)


class PolicyClause(StrictModel):
    clause_id: Annotated[str, StringConstraints(pattern=r"^POL-[A-Z]+-\d{3}$")]
    title: str
    category: str
    text: str


class ToolCallRecord(StrictModel):
    tool: str
    attempt: Annotated[int, Field(ge=1)]
    arguments: dict[str, object]
    success: bool
    error: str | None = None
    duration_ms: Annotated[float, Field(ge=0)]


# ---------------------------------------------------------------------------
# Investigation & planning
# ---------------------------------------------------------------------------
class Finding(StrictModel):
    kind: FindingKind
    transaction_ids: list[TransactionId] = Field(default_factory=list)
    detail: str


class RiskAssessment(StrictModel):
    score: Annotated[float, Field(ge=0.0, le=1.0)]
    level: RiskLevel
    flags: list[str] = Field(default_factory=list)


class RefundInstruction(StrictModel):
    transaction_id: TransactionId
    amount: Annotated[float, Field(gt=0, le=1_000_000)]


class ResolutionAction(StrictModel):
    action_type: ActionType
    refunds: list[RefundInstruction] = Field(default_factory=list)
    currency: CurrencyCode = "USD"
    reason_code: Annotated[str, StringConstraints(pattern=r"^[A-Z_]{3,40}$")]
    policy_refs: list[str] = Field(default_factory=list)
    secondary_actions: list[SecondaryAction] = Field(default_factory=list)
    customer_facing_reason: Annotated[str, StringConstraints(min_length=1, max_length=600)]

    @property
    def total_amount(self) -> float:
        return round(sum(r.amount for r in self.refunds), 2)

    @model_validator(mode="after")
    def _refund_consistency(self) -> ResolutionAction:
        if self.action_type == "REFUND" and not self.refunds:
            raise ValueError("REFUND action requires at least one refund instruction")
        if self.action_type != "REFUND" and self.refunds:
            raise ValueError(f"{self.action_type} action must not carry refund instructions")
        ids = [r.transaction_id for r in self.refunds]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate transaction in refund instructions")
        return self


class HumanDecision(StrictModel):
    approved: bool
    reviewer: Annotated[str, StringConstraints(min_length=2, max_length=80)]
    reason: Annotated[str, StringConstraints(max_length=500)] | None = None

    @model_validator(mode="after")
    def _reason_on_reject(self) -> HumanDecision:
        if not self.approved and not self.reason:
            raise ValueError("a rejection reason is required when approved=false")
        return self


class RefundReceipt(StrictModel):
    refund_id: str
    transaction_id: TransactionId
    amount: float
    currency: CurrencyCode
    status: Literal["succeeded", "duplicate_ignored"]
    idempotency_key: str


class ExecutionReceipt(StrictModel):
    executed: bool
    refunds: list[RefundReceipt] = Field(default_factory=list)
    secondary_actions_done: list[SecondaryAction] = Field(default_factory=list)
    note: str = ""

    @property
    def total_refunded(self) -> float:
        return round(sum(r.amount for r in self.refunds), 2)


# ---------------------------------------------------------------------------
# API payloads
# ---------------------------------------------------------------------------
class TicketCreate(StrictModel):
    message: Annotated[str, StringConstraints(min_length=10, max_length=6000)]
    customer_email: str | None = None

    @field_validator("customer_email", mode="before")
    @classmethod
    def _email(cls, v: object) -> str | None:
        if v is None or v == "":
            return None
        if not isinstance(v, str):
            raise ValueError("customer_email must be a string")
        return normalise_email(v)


class ChannelMeta(StrictModel):
    """Where a ticket came from. Email tickets are answered in the sender's thread."""

    channel: Literal["web", "email"] = "web"
    inbound_source: Literal["webhook", "imap", "simulator"] | None = None
    sender_email: str | None = None
    subject: Annotated[str, StringConstraints(max_length=500)] | None = None
    message_id: Annotated[str, StringConstraints(max_length=500)] | None = None


class TicketCreated(StrictModel):
    ticket_id: TicketId
    execution_status: ExecutionStatus


class DecisionRequest(HumanDecision):
    pass


class TicketView(BaseModel):
    """Read model returned by the API (not strict: it mirrors persisted state)."""

    model_config = ConfigDict(extra="ignore")

    ticket_id: str
    raw_message: str
    execution_status: str
    customer_email: str | None = None
    dispute_amount: float | None = None
    claim_category: str | None = None
    customer_tier: str | None = None
    requires_human_approval: bool = False
    human_approved: bool | None = None
    rejection_reason: str | None = None
    tool_error_count: int = 0
    investigation_summary: str = ""
    policy_clauses: list[str] = Field(default_factory=list)
    db_records: list[dict[str, object]] = Field(default_factory=list)
    findings: list[dict[str, object]] = Field(default_factory=list)
    risk: dict[str, object] | None = None
    resolution_action: dict[str, object] | None = None
    execution_receipt: dict[str, object] | None = None
    customer_response_draft: str = ""
    judge_verdict: dict[str, object] | None = None
    injection_signals: list[str] = Field(default_factory=list)
    tool_calls: list[dict[str, object]] = Field(default_factory=list)
    logs: list[str] = Field(default_factory=list)
    email_dispatched: bool = False
    channel: str = "web"
    inbound_source: str | None = None
    sender_email: str | None = None
    email_subject: str | None = None
    next_nodes: list[str] = Field(default_factory=list)
    pending_interrupt: dict[str, object] | None = None
    created_at: str | None = None
    updated_at: str | None = None
