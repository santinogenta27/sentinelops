"""LangGraph node implementations.

Every node:
  * reads typed data by re-validating state through Pydantic models,
  * returns only its state *delta* (``logs``/``tool_calls`` are append-only reducers),
  * has an LLM path and a deterministic fallback path (graceful degradation).
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from langgraph.types import interrupt
from pydantic import ValidationError

from app.agent import heuristics, prompts
from app.agent.autopilot import AUTOPILOT_REVIEWER
from app.agent.engine import (
    MAX_LOOKBACK_DAYS,
    assess_risk,
    inflated_amount_requested,
    investigate,
    plan_resolution,
    requires_human_approval,
)
from app.agent.guardrails import detect_prompt_injection, fence_untrusted, find_emails, find_money_amounts
from app.config import Settings
from app.db.database import utcnow
from app.db.repository import Repository
from app.integrations.crm_mock import CRMMock
from app.integrations.mailer import Mailer
from app.integrations.faults import FaultInjector, ToolError
from app.integrations.stripe_live import StripeLive
from app.integrations.stripe_mock import StripeMock
from app.llm.base import LLMClient, LLMError, LLMTask
from app.models.schemas import (
    ClaimCategory,
    CustomerDraft,
    CustomerRecord,
    ExecutionReceipt,
    Finding,
    HumanDecision,
    InvestigationNarrative,
    JudgeVerdict,
    RefundReceipt,
    ResolutionAction,
    RiskAssessment,
    SecondaryAction,
    ToolCallRecord,
    TransactionQuery,
    TransactionRecord,
    TriageExtraction,
)
from app.models.state import AgentState
from app.rag.retriever import PolicyRetriever

logger = logging.getLogger("sentinelops.agent")

DB_TOOL = "db.query_transactions"
NodeResult = dict[str, Any]
_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


@dataclass(slots=True)
class AgentDeps:
    settings: Settings
    repo: Repository
    retriever: PolicyRetriever
    stripe: StripeMock
    crm: CRMMock
    mailer: Mailer
    faults: FaultInjector
    llm: LLMClient | None
    clock: Callable[[], datetime] = field(default=utcnow)
    stripe_live: StripeLive | None = None  # real Stripe (test mode) mirror; None = ledger only


def _log(node: str, message: str) -> str:
    return f"{datetime.now().strftime('%H:%M:%S')} [{node}] {message}"


def _tool_record(tool: str, attempt: int, args: dict[str, object], started: float, error: str | None) -> dict[str, Any]:
    return ToolCallRecord(
        tool=tool,
        attempt=attempt,
        arguments=args,
        success=error is None,
        error=error,
        duration_ms=round((time.perf_counter() - started) * 1000, 2),
    ).model_dump(mode="json")


class SentinelNodes:
    def __init__(self, deps: AgentDeps) -> None:
        self.d = deps

    # ------------------------------------------------------------------ helpers
    @property
    def _llm(self) -> LLMClient | None:
        llm = self.d.llm
        return llm if llm is not None and llm.breaker.allow() else None

    @staticmethod
    def _customer(state: AgentState) -> CustomerRecord | None:
        profile = state.get("customer_profile")
        return CustomerRecord.model_validate(profile) if profile else None

    @staticmethod
    def _sender_mismatch(state: AgentState, customer: CustomerRecord | None) -> bool:
        """Email channel: the request came from an address that does not own the account."""
        sender = state.get("sender_email")
        if not sender or customer is None:
            return False
        if state.get("sender_alias_of") == customer.email:
            return False  # operator-verified alias of exactly this account
        return customer.email != sender

    @staticmethod
    def _transactions(state: AgentState) -> list[TransactionRecord]:
        return [TransactionRecord.model_validate(r) for r in state.get("db_records", [])]

    # =================================================================== TRIAGE
    async def triage(self, state: AgentState) -> NodeResult:
        node = "triage"
        raw = state["raw_message"]
        provided_email = state.get("customer_email")
        logs = [_log(node, f"received ticket ({len(raw)} chars)")]

        signals = detect_prompt_injection(raw)
        if signals:
            logs.append(_log(node, f"⚠ prompt-injection signatures detected: {signals} → message treated as hostile data"))

        extraction: TriageExtraction | None = None
        llm = self._llm
        if llm is not None:
            try:
                extraction = await llm.generate_structured(
                    task=LLMTask.TRIAGE,
                    system=prompts.TRIAGE_SYSTEM,
                    prompt=fence_untrusted(raw),
                    schema=TriageExtraction,
                )
                logs.append(_log(node, f"LLM extraction via {llm.provider}:{llm.model}"))
            except LLMError as exc:
                logs.append(_log(node, f"LLM unavailable ({str(exc)[:120]}) → deterministic extractor"))
        if extraction is None:
            extraction = heuristics.heuristic_triage(raw, provided_email)
            logs.append(_log(node, "deterministic extraction"))

        # --- Anti-hallucination grounding of extracted entities ---------------
        literal_emails = set(find_emails(raw))
        email = provided_email
        if email is None and extraction.customer_email in literal_emails:
            email = extraction.customer_email
        elif email is None and extraction.customer_email is not None:
            logs.append(_log(node, f"dropped hallucinated email {extraction.customer_email!r} (not in message)"))
        literal_numbers = {round(float(n.replace(",", "")), 2) for n in _NUMBER_RE.findall(raw) if n.replace(",", "").replace(".", "", 1).isdigit()}
        amount = extraction.dispute_amount
        if amount is not None and all(abs(amount - n) >= 0.005 for n in literal_numbers):
            logs.append(_log(node, f"dropped hallucinated amount {amount} (not in message)"))
            amount = None
        refs = [r for r in extraction.transaction_refs if r.lower() in raw.lower()]

        logs.append(
            _log(node, f"category={extraction.claim_category} email={email} amount={amount} refs={refs}")
        )
        return {
            "customer_email": email,
            "dispute_amount": amount,
            "claim_category": extraction.claim_category,
            "transaction_refs": refs,
            "injection_signals": signals,
            "investigation_summary": extraction.summary,
            "execution_status": "INVESTIGATING",
            "logs": logs,
        }

    # ============================================================== DB INSPECTOR
    async def _repair_args(self, raw_args: dict[str, Any], error: str, state: AgentState) -> tuple[TransactionQuery | None, str]:
        llm = self._llm
        if llm is not None:
            try:
                fixed = await llm.generate_structured(
                    task=LLMTask.REPAIR_TOOL_ARGS,
                    system=prompts.REPAIR_SYSTEM,
                    prompt=(
                        f"Tool: query_transactions\nPrevious arguments: {json.dumps(raw_args)}\nError: {error}\n"
                        f"Known customer email: {state.get('customer_email')}\n{fence_untrusted(state['raw_message'])}"
                    ),
                    schema=TransactionQuery,
                )
                return fixed, f"LLM repair ({llm.provider})"
            except LLMError:
                pass
        return heuristics.repair_query_args(raw_args, state.get("customer_email")), "deterministic repair"

    async def db_inspector(self, state: AgentState) -> NodeResult:
        node = "db_inspector"
        attempt = state["tool_error_count"] + 1
        logs: list[str] = []
        prior_error = state.get("last_tool_error")
        raw_args: dict[str, Any] = dict(state.get("tool_query_args") or {})
        if not raw_args:
            raw_args = {
                "customer_email": state.get("customer_email"),
                "transaction_ids": list(state.get("transaction_refs", [])),
                "lookback_days": 90,
            }
        if not raw_args.get("customer_email") and not raw_args.get("transaction_ids"):
            logs.append(_log(node, "no customer identifiers in ticket → skipping ledger query"))
            return {
                "db_records": [],
                "customer_profile": None,
                "last_tool_error": None,
                "requery_requested": False,
                "execution_status": "INVESTIGATING",
                "logs": logs,
            }

        started = time.perf_counter()
        if prior_error and prior_error.startswith("VALIDATION"):
            repaired, how = await self._repair_args(raw_args, prior_error, state)
            logs.append(_log(node, f"self-correction attempt {attempt}: {how} → {repaired.model_dump() if repaired else 'unrepairable'}"))
            if repaired is None:
                return {
                    "db_records": [],
                    "customer_profile": None,
                    "last_tool_error": None,
                    "requery_requested": False,
                    "logs": logs + [_log(node, "arguments unrepairable → continuing without ledger data")],
                }
            raw_args = repaired.model_dump(mode="json")
        elif prior_error:
            logs.append(_log(node, f"self-correction attempt {attempt}: retrying after transient error"))

        try:
            query = TransactionQuery.model_validate(raw_args)
        except ValidationError as exc:
            err = "VALIDATION: " + "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
            logs.append(_log(node, f"✗ invalid tool arguments (attempt {attempt}): {err}"))
            return {
                "tool_error_count": state["tool_error_count"] + 1,
                "last_tool_error": err,
                "tool_query_args": raw_args,
                "tool_calls": [_tool_record(DB_TOOL, attempt, raw_args, started, err)],
                "logs": logs,
            }

        args_json = query.model_dump(mode="json")
        try:
            self.d.faults.check(DB_TOOL)
            customer, txns = await self.d.repo.query_transactions(query)
        except ToolError as exc:
            err = f"TOOL: {exc}"
            logs.append(_log(node, f"✗ tool failure (attempt {attempt}): {exc}"))
            return {
                "tool_error_count": state["tool_error_count"] + 1,
                "last_tool_error": err,
                "tool_query_args": args_json,
                "tool_calls": [_tool_record(DB_TOOL, attempt, args_json, started, err)],
                "logs": logs,
            }

        logs.append(
            _log(
                node,
                f"✓ {DB_TOOL}(lookback={query.lookback_days}d) → customer={customer.id if customer else None} "
                f"tier={customer.tier if customer else None} transactions={len(txns)}",
            )
        )
        return {
            "db_records": [t.model_dump(mode="json") for t in txns],
            "customer_profile": customer.model_dump(mode="json") if customer else None,
            "customer_tier": customer.tier if customer else None,
            "customer_email": state.get("customer_email") or (customer.email if customer else None),
            "last_tool_error": None,
            "requery_requested": False,
            "tool_query_args": args_json,
            "tool_calls": [_tool_record(DB_TOOL, attempt, args_json, started, None)],
            "logs": logs,
        }

    # ========================================================== POLICY RETRIEVER
    async def policy_retriever(self, state: AgentState) -> NodeResult:
        node = "policy_retriever"
        started = time.perf_counter()
        category = state.get("claim_category") or "GENERAL"
        query = f"{category.replace('_', ' ').lower()} {state.get('investigation_summary', '')}"[:1000]
        hits = self.d.retriever.search(query, k=4, category_hint=category)
        clauses = [h.render() for h in hits]
        if state.get("injection_signals"):
            sec = self.d.retriever.get("POL-SEC-001")
            if sec and all(not c.startswith("[POL-SEC-001]") for c in clauses):
                clauses.append(f"[{sec.clause_id}] {sec.title}: {sec.text}")
        return {
            "policy_clauses": clauses,
            "execution_status": "PLANNING",
            "tool_calls": [
                _tool_record("rag.hybrid_search", 1, {"query": query[:200], "k": 4, "category": category}, started, None)
            ],
            "logs": [_log(node, "hybrid (Chroma dense + BM25, RRF) → " + ", ".join(f"{h.clause.clause_id}({h.score})" for h in hits))],
        }

    # ======================================================== RESOLUTION PLANNER
    async def resolution_planner(self, state: AgentState) -> NodeResult:
        node = "resolution_planner"
        s = self.d.settings
        now = self.d.clock()
        customer = self._customer(state)
        txns = self._transactions(state)
        category: ClaimCategory = state.get("claim_category") or "GENERAL"  # type: ignore[assignment]
        args = state.get("tool_query_args") or {}
        referenced = [r for r in args.get("transaction_ids", []) if isinstance(r, str)]
        lookback = int(args.get("lookback_days", 90))
        tool_failure = state.get("last_tool_error") is not None
        raw = state["raw_message"]
        claimed = [a for a in find_money_amounts(raw) if a <= 1_000_000]
        refund_requested = heuristics.requests_refund(raw)
        signals = list(state.get("injection_signals", []))
        if refund_requested and inflated_amount_requested(txns, claimed) and "unsupported_amount" not in signals:
            # Ledger-aware check (POL-SEC-001): asking for more than the cited charge is manipulation.
            signals.append("unsupported_amount")
        injection = bool(signals)

        inv = investigate(
            customer=customer,
            txns=txns,
            category=category,
            referenced_ids=referenced,
            dispute_amount=state.get("dispute_amount"),
            claimed_amounts=claimed,
            now=now,
        )
        plan = plan_resolution(
            inv=inv,
            category=category,
            lookback_days=lookback,
            tool_failure=tool_failure,
            injection_detected=injection,
            refund_requested=refund_requested,
        )
        if plan.action is None:
            new_args = {**args, "lookback_days": plan.requery_lookback_days or MAX_LOOKBACK_DAYS}
            return {
                "requery_requested": True,
                "tool_query_args": new_args,
                "logs": [
                    _log(
                        node,
                        f"referenced charges {inv.missing_references} not in {lookback}-day window → "
                        f"re-querying DBInspector with lookback={new_args['lookback_days']}d",
                    )
                ],
            }

        action = plan.action
        sender_mismatch = self._sender_mismatch(state, customer)
        if sender_mismatch and "POL-ID-001" not in action.policy_refs:
            action = action.model_copy(update={"policy_refs": [*action.policy_refs, "POL-ID-001"]})
        risk = assess_risk(
            customer=customer,
            findings=inv.findings,
            category=category,
            injection_signals=signals,
            payout_total=action.total_amount,
            hitl_threshold=s.hitl_threshold_usd,
            now=now,
            sender_mismatch=sender_mismatch,
        )
        needs_human = requires_human_approval(
            action, risk, injection_detected=injection, hitl_threshold=s.hitl_threshold_usd
        )
        clauses = list(state.get("policy_clauses", []))
        for pid in action.policy_refs:
            pol = self.d.retriever.get(pid)
            if pol and all(not c.startswith(f"[{pid}]") for c in clauses):
                clauses.append(f"[{pol.clause_id}] {pol.title}: {pol.text}")

        summary = await self._narrative(state, customer, inv.findings, action, risk, needs_human)
        logs = [
            _log(node, f"findings: {[f.kind for f in inv.findings]}"),
            _log(
                node,
                f"plan: {action.action_type} {action.reason_code} total=${action.total_amount:.2f} "
                f"policies={action.policy_refs} secondary={action.secondary_actions}",
            ),
            _log(node, f"risk: {risk.level} ({risk.score}) flags={risk.flags}"),
            _log(
                node,
                "→ HUMAN APPROVAL REQUIRED (HITL gate)" if needs_human else "→ auto-approved (≤ threshold, low risk)",
            ),
        ]
        return {
            "findings": [f.model_dump(mode="json") for f in inv.findings],
            "risk": risk.model_dump(mode="json"),
            "resolution_action": action.model_dump(mode="json"),
            "requires_human_approval": needs_human,
            "injection_signals": signals,
            "policy_clauses": clauses,
            "investigation_summary": summary,
            "requery_requested": False,
            "execution_status": "AWAITING_APPROVAL" if needs_human else "EXECUTING",
            "logs": logs,
        }

    async def _narrative(
        self,
        state: AgentState,
        customer: CustomerRecord | None,
        findings: list[Finding],
        action: ResolutionAction,
        risk: RiskAssessment,
        needs_human: bool,
    ) -> str:
        customer_line = (
            f"Customer {customer.name} <{customer.email}> ({customer.tier}, {customer.country}, "
            f"{customer.chargeback_count} prior chargebacks)."
            if customer
            else "Customer could not be identified."
        )
        facts = {
            "complaint_summary": state.get("investigation_summary", ""),
            "claim_category": state.get("claim_category"),
            "claimed_amount": state.get("dispute_amount"),
            "customer": customer.model_dump(mode="json") if customer else None,
            "findings": [f.model_dump(mode="json") for f in findings],
            "proposed_action": action.model_dump(mode="json"),
            "total_payout": action.total_amount,
            "risk": risk.model_dump(mode="json"),
            "injection_signals": state.get("injection_signals", []),
            "requires_human_approval": needs_human,
        }
        llm = self._llm
        if llm is not None:
            try:
                out = await llm.generate_structured(
                    task=LLMTask.INVESTIGATION_SUMMARY,
                    system=prompts.SUMMARY_SYSTEM,
                    prompt=f"FACTS:\n{json.dumps(facts, default=str)}",
                    schema=InvestigationNarrative,
                )
                return out.summary
            except LLMError:
                pass
        return heuristics.template_narrative(
            customer_line=customer_line,
            findings=[f"{f.kind}: {f.detail}" for f in findings],
            plan_line=f"{action.action_type} ({action.reason_code}) total ${action.total_amount:,.2f} — {', '.join(action.policy_refs)}",
            risk_line=f"{risk.level} score={risk.score} flags={', '.join(risk.flags) or 'none'}",
            approval_line="Human authorisation REQUIRED before execution." if needs_human else "Within autonomous authority.",
        )

    # ================================================================ HUMAN GATE
    async def human_gate(self, state: AgentState) -> NodeResult:
        """Pauses the graph with ``interrupt``. Pure until the interrupt: safe to re-run on resume."""
        node = "human_gate"
        action = ResolutionAction.model_validate(state["resolution_action"])
        raw_decision = interrupt(
            {
                "type": "approval_request",
                "ticket_id": state["ticket_id"],
                "amount": action.total_amount,
                "currency": action.currency,
                "action": action.model_dump(mode="json"),
                "risk": state.get("risk"),
                "injection_signals": state.get("injection_signals", []),
                "investigation_summary": state.get("investigation_summary", ""),
                "question": f"Authorise {action.action_type} of ${action.total_amount:,.2f} {action.currency}?",
            }
        )
        decision = HumanDecision.model_validate(raw_decision)
        verdict = "APPROVED" if decision.approved else f"REJECTED ({decision.reason})"
        who = (
            "⚡ Auto-Pilot (policy-compliant, zero-touch)"
            if decision.reviewer == AUTOPILOT_REVIEWER
            else f"supervisor {decision.reviewer}"
        )
        return {
            "human_approved": decision.approved,
            "reviewer": decision.reviewer,
            "rejection_reason": None if decision.approved else decision.reason,
            "execution_status": "EXECUTING" if decision.approved else "DRAFTING",
            "logs": [_log(node, f"{who} → {verdict}")],
        }

    # ============================================================ ACTION EXECUTOR
    async def action_executor(self, state: AgentState) -> NodeResult:
        node = "action_executor"
        action = ResolutionAction.model_validate(state["resolution_action"])
        logs: list[str] = []
        tool_calls: list[dict[str, Any]] = []

        # Defence in depth: never trust routing alone for money movement.
        if state.get("requires_human_approval") and state.get("human_approved") is not True:
            receipt = ExecutionReceipt(executed=False, note="BLOCKED: human approval required but not granted")
            logs.append(_log(node, "⛔ execution blocked — missing human authorisation"))
            return {"execution_receipt": receipt.model_dump(mode="json"), "execution_status": "DRAFTING", "logs": logs}
        if action.action_type == "REFUND" and action.total_amount > self.d.settings.hitl_threshold_usd and state.get("human_approved") is not True:
            receipt = ExecutionReceipt(executed=False, note="BLOCKED: amount above autonomous threshold")
            logs.append(_log(node, "⛔ execution blocked — amount above autonomous authority"))
            return {"execution_receipt": receipt.model_dump(mode="json"), "execution_status": "DRAFTING", "logs": logs}

        receipts: list[RefundReceipt] = []
        failures: list[str] = []
        approver = state.get("reviewer") or "sentinelops-autopilot"
        for refund in action.refunds:
            key = f"{state['ticket_id']}:{refund.transaction_id}"
            args: dict[str, object] = {"transaction_id": refund.transaction_id, "amount": refund.amount, "idempotency_key": key}
            for attempt in range(1, self.d.settings.max_tool_retries + 2):
                started = time.perf_counter()
                try:
                    rec = await self.d.stripe.create_refund(
                        transaction_id=refund.transaction_id,
                        amount=refund.amount,
                        currency=action.currency,
                        idempotency_key=key,
                        ticket_id=state["ticket_id"],
                        approved_by=approver,
                    )
                    receipts.append(rec)
                    tool_calls.append(_tool_record(StripeMock.TOOL_NAME, attempt, args, started, None))
                    logs.append(_log(node, f"✓ refund {rec.refund_id} ${rec.amount:.2f} on {rec.transaction_id} ({rec.status})"))
                    break
                except ToolError as exc:
                    tool_calls.append(_tool_record(StripeMock.TOOL_NAME, attempt, args, started, str(exc)))
                    logs.append(_log(node, f"✗ stripe attempt {attempt}: {exc}"))
                    if not exc.retryable or attempt > self.d.settings.max_tool_retries:
                        failures.append(f"{refund.transaction_id}: {exc}")
                        break

        done: list[SecondaryAction] = []
        customer = self._customer(state)
        txns = {t.id: t for t in self._transactions(state)}
        for sec in action.secondary_actions:
            try:
                if sec == "FREEZE_CARD" and customer:
                    for r in action.refunds:
                        if r.transaction_id in txns:
                            await self.d.stripe.freeze_card(
                                card_last4=txns[r.transaction_id].card_last4, customer_id=customer.id, ticket_id=state["ticket_id"]
                            )
                    done.append(sec)
                elif sec == "FLAG_ACCOUNT":
                    await self.d.crm.add_note(
                        customer_id=customer.id if customer else None,
                        ticket_id=state["ticket_id"],
                        note=f"Trust & Safety flag: {(state.get('risk') or {}).get('flags')}",
                    )
                    done.append(sec)
                elif sec == "RECONCILE_GATEWAY":
                    await self.d.crm.add_note(
                        customer_id=customer.id if customer else None,
                        ticket_id=state["ticket_id"],
                        note=f"Gateway reconciliation job opened for {[r.transaction_id for r in action.refunds]}",
                    )
                    done.append(sec)
            except ToolError as exc:
                logs.append(_log(node, f"secondary action {sec} failed (non-fatal): {exc}"))
        if done:
            logs.append(_log(node, f"secondary actions done: {done}"))

        mirror = await self._mirror_to_stripe(state, action, receipts, approver, logs, tool_calls)

        executed = not failures
        if action.action_type != "REFUND":
            logs.append(_log(node, f"no payout for {action.action_type}"))
        receipt = ExecutionReceipt(
            executed=executed,
            refunds=receipts,
            secondary_actions_done=done,
            note="; ".join(failures) if failures else "ok",
        )
        return {
            "execution_receipt": receipt.model_dump(mode="json"),
            "execution_status": "DRAFTING",
            "stripe_mirror": mirror,
            "tool_calls": tool_calls,
            "logs": logs,
        }

    async def _mirror_to_stripe(
        self,
        state: AgentState,
        action: ResolutionAction,
        receipts: list[RefundReceipt],
        approver: str,
        logs: list[str],
        tool_calls: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Mirror ledger refunds into Stripe TEST mode (observability only — never affects the outcome)."""
        live = self.d.stripe_live
        if live is None or not live.mirror_enabled or not receipts:
            return list(state.get("stripe_mirror") or [])
        node = "action_executor"
        customer = self._customer(state)
        txns = {t.id: t for t in self._transactions(state)}
        out: list[dict[str, Any]] = []
        for rec in receipts:
            txn = txns.get(rec.transaction_id) or await self.d.repo.get_transaction(rec.transaction_id)
            started = time.perf_counter()
            result = await live.mirror_refund(
                local_refund_id=rec.refund_id,
                ticket_id=state["ticket_id"],
                transaction_id=rec.transaction_id,
                transaction_amount=txn.amount if txn else rec.amount,
                amount=rec.amount,
                currency=rec.currency,
                reason_code=action.reason_code,
                customer_email=customer.email if customer else None,
                customer_name=customer.name if customer else None,
                approved_by=approver,
            )
            args: dict[str, object] = {"ledger_refund_id": rec.refund_id, "amount": rec.amount, "mode": live.mode}
            tool_calls.append(
                _tool_record(StripeLive.MIRROR_TOOL, 1, args, started, result.error if result.status == "failed" else None)
            )
            if result.status in ("mirrored", "duplicate"):
                logs.append(_log(node, f"✓ Stripe test mode: refund {result.stripe_refund_id} on {result.payment_intent_id} ({result.status})"))
            else:
                logs.append(_log(node, f"Stripe mirror {result.status} (non-fatal): {result.error}"))
            out.append({"local_refund_id": rec.refund_id, "transaction_id": rec.transaction_id, "amount": rec.amount,
                        "currency": rec.currency, **result.as_dict()})
        return out

    # ========================================================= RESPONSE DRAFTER
    def _facts(self, state: AgentState) -> dict[str, Any]:
        action = ResolutionAction.model_validate(state["resolution_action"])
        receipt = ExecutionReceipt.model_validate(state["execution_receipt"]) if state.get("execution_receipt") else None
        customer = self._customer(state)
        executed: list[dict[str, Any]] = [
            {"transaction_id": r.transaction_id, "amount": r.amount, "currency": r.currency}
            for r in (receipt.refunds if receipt else [])
        ]
        return {
            "ticket_reference": state["ticket_id"],
            # Never reveal the account holder's name to an unverified sender (POL-ID-001).
            "customer_name": customer.name if customer and not self._sender_mismatch(state, customer) else None,
            "outcome": action.action_type,
            "human_rejected": state.get("human_approved") is False,
            "execution_failed": bool(receipt and not receipt.executed and action.action_type == "REFUND"),
            "executed_refunds": executed,
            "total_refunded": round(sum(float(e["amount"]) for e in executed), 2),
            "customer_facing_reason": action.customer_facing_reason,
            "customer_transaction_ids": [t.id for t in self._transactions(state)],
        }

    def _template(self, state: AgentState) -> CustomerDraft:
        f = self._facts(state)
        return heuristics.template_draft(
            customer_name=f["customer_name"],
            ticket_id=state["ticket_id"],
            action_type=f["outcome"],
            executed_refunds=[(e["transaction_id"], e["amount"]) for e in f["executed_refunds"]],
            customer_facing_reason=f["customer_facing_reason"],
            human_rejected=f["human_rejected"],
            execution_failed=f["execution_failed"],
        )

    async def response_drafter(self, state: AgentState) -> NodeResult:
        node = "response_drafter"
        attempt = state.get("draft_attempts", 0) + 1
        facts = self._facts(state)
        previous = state.get("judge_verdict")
        draft: CustomerDraft | None = None
        llm = self._llm
        if llm is not None:
            feedback = ""
            if previous and not previous.get("passed"):
                feedback = (
                    f"\n\nYOUR PREVIOUS DRAFT WAS REJECTED BY THE AUDITOR.\nPrevious draft:\n{state.get('customer_response_draft')}\n"
                    f"Auditor feedback: {previous.get('feedback')}\nIssues: {previous.get('issues')}\nFix every issue."
                )
            try:
                draft = await llm.generate_structured(
                    task=LLMTask.DRAFT_RESPONSE,
                    system=prompts.DRAFT_SYSTEM,
                    prompt=f"FACTS:\n{json.dumps(facts)}{feedback}",
                    schema=CustomerDraft,
                )
            except LLMError as exc:
                logger.info("drafter falling back to template: %s", exc)
        source = f"LLM ({llm.provider})" if draft is not None and llm is not None else "grounded template"
        if draft is None:
            draft = self._template(state)
        return {
            "customer_response_draft": f"Subject: {draft.subject}\n\n{draft.body}",
            "draft_attempts": attempt,
            "execution_status": "AUDITING",
            "logs": [_log(node, f"draft #{attempt} written by {source}" + (" (re-draft after judge feedback)" if attempt > 1 else ""))],
        }

    # ================================================================ LLM JUDGE
    def _grounding_sets(self, state: AgentState) -> tuple[set[float], set[str], set[float]]:
        f = self._facts(state)
        executed_amounts = {round(e["amount"], 2) for e in f["executed_refunds"]}
        allowed = executed_amounts | ({f["total_refunded"]} if executed_amounts else set())
        required = {f["total_refunded"]} if executed_amounts else set()
        return allowed, set(f["customer_transaction_ids"]), required

    async def llm_judge(self, state: AgentState) -> NodeResult:
        node = "llm_judge"
        s = self.d.settings
        draft = state.get("customer_response_draft", "")
        allowed, ids, required = self._grounding_sets(state)
        det = heuristics.heuristic_judge(
            draft, allowed_amounts=allowed, allowed_transaction_ids=ids, required_amounts=required, threshold=s.judge_pass_threshold
        )
        verdict = det
        source = "deterministic grounding audit"
        llm = self._llm
        if llm is not None:
            try:
                llm_verdict = await llm.generate_structured(
                    task=LLMTask.JUDGE,
                    system=prompts.JUDGE_SYSTEM,
                    prompt=f"FACTS:\n{json.dumps(self._facts(state))}\n\nDRAFT:\n{draft}",
                    schema=JudgeVerdict,
                )
                score = round(min(llm_verdict.faithfulness_score, det.faithfulness_score), 2)
                verdict = JudgeVerdict(
                    passed=score >= s.judge_pass_threshold and llm_verdict.passed and det.passed,
                    faithfulness_score=score,
                    feedback=f"{llm_verdict.feedback} | {det.feedback}"[:1500],
                    issues=(llm_verdict.issues + det.issues)[:20],
                )
                source = f"LLM judge ({llm.provider}) + deterministic grounding audit"
            except LLMError:
                pass

        logs = [_log(node, f"{source}: score={verdict.faithfulness_score} passed={verdict.passed}")]
        update: NodeResult = {"judge_verdict": verdict.model_dump(mode="json")}
        if not verdict.passed:
            if state.get("draft_attempts", 1) <= s.max_redrafts:
                logs.append(_log(node, f"score < {s.judge_pass_threshold} → re-draft requested: {verdict.issues}"))
            else:
                safe = self._template(state)
                safe_text = f"Subject: {safe.subject}\n\n{safe.body}"
                safe_verdict = heuristics.heuristic_judge(
                    safe_text, allowed_amounts=allowed, allowed_transaction_ids=ids, required_amounts=required,
                    threshold=s.judge_pass_threshold,
                )
                update["customer_response_draft"] = safe_text
                update["judge_verdict"] = safe_verdict.model_copy(
                    update={"feedback": f"Re-draft budget exhausted; replaced with grounded template. {safe_verdict.feedback}"}
                ).model_dump(mode="json")
                logs.append(_log(node, "re-draft budget exhausted → grounded safe template substituted"))
        update["logs"] = logs
        return update

    # ================================================================= DISPATCH
    async def dispatch(self, state: AgentState) -> NodeResult:
        node = "dispatch"
        action = ResolutionAction.model_validate(state["resolution_action"])
        receipt = ExecutionReceipt.model_validate(state["execution_receipt"]) if state.get("execution_receipt") else None
        draft = state.get("customer_response_draft", "")
        subject, _, body = draft.partition("\n\n")
        subject = subject.removeprefix("Subject: ").strip() or "Update on your billing request"

        if state.get("human_approved") is False:
            final = "REJECTED"
        elif action.action_type == "REFUND" and receipt is not None and not receipt.executed:
            final = "FAILED"
        elif action.action_type in ("ESCALATE", "REQUEST_INFO"):
            final = "ESCALATED"
        else:
            final = "EXECUTED"

        logs: list[str] = []
        delivery = "not sent"
        customer = self._customer(state)
        channel = state.get("channel") or "web"
        original_subject = state.get("email_subject")
        if channel == "email" and state.get("sender_email"):
            # Reply in the customer's own email thread.
            to_addr = state.get("sender_email")
            base = original_subject or "Your billing request"
            mail_subject = base if base.lower().startswith("re:") else f"Re: {base}"
        else:
            to_addr = customer.email if customer else state.get("customer_email")
            mail_subject = subject
        try:
            delivery = await self.d.mailer.send_customer_reply(
                ticket_id=state["ticket_id"],
                to_address=to_addr,
                subject=mail_subject,
                heading=subject,
                body=body,
                channel=channel,
                inbound_source=state.get("inbound_source"),
                in_reply_to=state.get("email_message_id"),
                # Real SMTP only to an identity we know: the account owner or an operator-verified alias.
                verified_recipient=customer is not None and not self._sender_mismatch(state, customer),
            )
            await self.d.crm.add_note(
                customer_id=customer.id if customer else None,
                ticket_id=state["ticket_id"],
                note=f"{final}: {action.action_type}/{action.reason_code} total=${action.total_amount:.2f} reviewer={state.get('reviewer')}",
            )
        except ToolError as exc:
            logs.append(_log(node, f"email/CRM dispatch failed: {exc}"))
        how = {
            "smtp": "delivered via SMTP",
            "outbox": "queued in outbox (simulated delivery)",
            "duplicate": "already sent (idempotent)",
            "smtp_failed": "SMTP delivery failed — kept in outbox for retry",
        }.get(delivery, delivery)
        logs.append(_log(node, f"email to {to_addr} {how} → status {final}"))
        return {"email_dispatched": True, "execution_status": final, "logs": logs}
