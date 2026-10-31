"""Application runtime: wires dependencies, drives graph runs, streams events, resumes HITL."""

from __future__ import annotations

import asyncio
import logging
import secrets
from collections.abc import AsyncGenerator
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

import aiosqlite
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.state import CompiledStateGraph
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command, StreamMode

from app.agent.graph import build_graph
from app.agent.nodes import AgentDeps
from app.config import Settings
from app.db.database import Database, iso, utcnow
from app.db.repository import Repository
from app.db.seed import seed_database
from app.integrations.crm_mock import CRMMock
from app.integrations.faults import FaultInjector, fault_injector
from app.integrations.stripe_mock import StripeMock
from app.llm.base import LLMClient
from app.llm.factory import build_llm
from app.models.schemas import TERMINAL_STATUSES, HumanDecision, TicketView
from app.models.state import AgentState, initial_state
from app.rag.embeddings import Embedder, GeminiEmbedder, HashingEmbedder
from app.rag.retriever import PolicyRetriever, load_policies

logger = logging.getLogger("sentinelops.runtime")

Event = dict[str, Any]
STREAM_MODES: list[StreamMode] = ["tasks", "updates"]


class EventBus:
    """In-process pub/sub fan-out for Server-Sent Events."""

    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[Event]] = set()

    def publish(self, event: Event) -> None:
        for q in list(self._subscribers):
            with suppress(asyncio.QueueFull):
                q.put_nowait(event)

    async def subscribe(self) -> AsyncGenerator[Event, None]:
        q: asyncio.Queue[Event] = asyncio.Queue(maxsize=1000)
        self._subscribers.add(q)
        try:
            while True:
                yield await q.get()
        finally:
            self._subscribers.discard(q)


class TicketNotFoundError(Exception):
    pass


class InvalidTransitionError(Exception):
    pass


@dataclass(slots=True)
class _Owned:
    conn: aiosqlite.Connection | None = None


def new_ticket_id() -> str:
    return f"TCK-{secrets.token_hex(4).upper()}"


def _summarise(values: dict[str, Any]) -> dict[str, object]:
    action = values.get("resolution_action") or {}
    refunds = action.get("refunds") or []
    return {
        "claim_category": values.get("claim_category"),
        "dispute_amount": values.get("dispute_amount"),
        "customer_tier": values.get("customer_tier"),
        "requires_human_approval": values.get("requires_human_approval", False),
        "human_approved": values.get("human_approved"),
        "action_type": action.get("action_type"),
        "payout_total": round(sum(float(r.get("amount", 0)) for r in refunds), 2),
        "risk_level": (values.get("risk") or {}).get("level"),
        "judge_score": (values.get("judge_verdict") or {}).get("faithfulness_score"),
        "injection": bool(values.get("injection_signals")),
    }


class SentinelRuntime:
    def __init__(
        self,
        *,
        settings: Settings,
        db: Database,
        deps: AgentDeps,
        graph: CompiledStateGraph[AgentState, None, AgentState, AgentState],
        owned: _Owned,
    ) -> None:
        self.settings = settings
        self.db = db
        self.deps = deps
        self.graph = graph
        self.bus = EventBus()
        self._owned = owned
        self._locks: dict[str, asyncio.Lock] = {}
        self._tasks: set[asyncio.Task[None]] = set()

    # ------------------------------------------------------------------ factory
    @classmethod
    async def create(
        cls,
        settings: Settings,
        *,
        persistent_checkpoints: bool = True,
        llm: LLMClient | None | str = "auto",
        faults: FaultInjector | None = None,
        persist_vectors: bool = True,
    ) -> SentinelRuntime:
        db = Database(settings.business_db_path)
        await db.init_schema()
        await seed_database(db)

        embedder: Embedder = HashingEmbedder()
        if settings.embedding_provider == "gemini" and settings.gemini_api_key:
            embedder = GeminiEmbedder(settings.gemini_api_key.get_secret_value(), settings.gemini_embedding_model)
        retriever = PolicyRetriever(
            load_policies(settings.policies_path),
            embedder,
            persist_path=settings.chroma_path if persist_vectors else None,
        )
        llm_client = build_llm(settings) if llm == "auto" else llm
        if isinstance(llm_client, str):
            raise ValueError("llm must be 'auto', None or an LLMClient")
        injector = faults or fault_injector
        deps = AgentDeps(
            settings=settings,
            repo=Repository(db),
            retriever=retriever,
            stripe=StripeMock(db, injector),
            crm=CRMMock(db, injector),
            faults=injector,
            llm=llm_client,
        )
        owned = _Owned()
        checkpointer: BaseCheckpointSaver[str]
        if persistent_checkpoints:
            owned.conn = await aiosqlite.connect(settings.checkpoint_db_path)
            saver = AsyncSqliteSaver(owned.conn)
            await saver.setup()
            checkpointer = saver
        else:
            checkpointer = InMemorySaver()
        graph = build_graph(deps, checkpointer)
        return cls(settings=settings, db=db, deps=deps, graph=graph, owned=owned)

    async def close(self) -> None:
        for t in list(self._tasks):
            t.cancel()
        for t in list(self._tasks):
            with suppress(asyncio.CancelledError, Exception):
                await t
        if self._owned.conn is not None:
            await self._owned.conn.close()

    def llm_status(self) -> dict[str, object]:
        if self.deps.llm is None:
            return {"provider": "deterministic", "model": None, "circuit": "n/a", "circuit_reason": None, "calls_made": 0}
        return self.deps.llm.status()

    # ----------------------------------------------------------------- running
    @staticmethod
    def _config(ticket_id: str) -> RunnableConfig:
        return {"configurable": {"thread_id": ticket_id}, "recursion_limit": 60}

    def _lock(self, ticket_id: str) -> asyncio.Lock:
        return self._locks.setdefault(ticket_id, asyncio.Lock())

    async def _emit(self, ticket_id: str, type_: str, node: str | None, payload: dict[str, object]) -> None:
        event_id = await self.db.append_event(ticket_id, type_, node, payload)
        self.bus.publish({"id": event_id, "ticket_id": ticket_id, "type": type_, "node": node, "ts": iso(utcnow()), "payload": payload})

    async def _drive(self, ticket_id: str, graph_input: AgentState | Command[Any] | None) -> dict[str, Any]:
        config = self._config(ticket_id)
        try:
            async for mode, chunk in self.graph.astream(graph_input, config, stream_mode=STREAM_MODES):
                if mode == "tasks" and isinstance(chunk, dict) and "input" in chunk:
                    await self._emit(ticket_id, "node_started", str(chunk.get("name")), {})
                elif mode == "updates" and isinstance(chunk, dict):
                    for node, delta in chunk.items():
                        if node == "__interrupt__":
                            interrupts = delta if isinstance(delta, (list, tuple)) else [delta]
                            for itr in interrupts:
                                value = getattr(itr, "value", itr)
                                await self._emit(ticket_id, "interrupt", "human_gate", {"request": value})
                        else:
                            await self._emit(ticket_id, "node_completed", node, {"delta": delta or {}})
        except Exception as exc:  # noqa: BLE001 - surface any graph failure to the operator
            logger.exception("graph run failed for %s", ticket_id)
            await self._emit(ticket_id, "run_error", None, {"error": f"{type(exc).__name__}: {exc}"})
            snap = await self.graph.aget_state(config)
            values = dict(snap.values)
            await self.db.upsert_ticket(ticket_id, values.get("raw_message", ""), values.get("customer_email"), "FAILED", _summarise(values))
            return values

        snap = await self.graph.aget_state(config)
        values = dict(snap.values)
        status = str(values.get("execution_status", "FAILED"))
        await self.db.upsert_ticket(ticket_id, values["raw_message"], values.get("customer_email"), status, _summarise(values))
        if snap.next:
            await self._emit(ticket_id, "paused", None, {"next": list(snap.next), "status": status})
        else:
            await self._emit(ticket_id, "run_completed", None, {"status": status})
        return values

    async def run_ticket(self, raw_message: str, customer_email: str | None = None, ticket_id: str | None = None) -> tuple[str, dict[str, Any]]:
        """Run a ticket until completion or the HITL interrupt (awaitable; used by tests and eval)."""
        tid = ticket_id or new_ticket_id()
        await self.db.upsert_ticket(tid, raw_message, customer_email, "TRIAGING")
        await self._emit(tid, "ticket_created", None, {"raw_message": raw_message, "customer_email": customer_email})
        async with self._lock(tid):
            values = await self._drive(tid, initial_state(tid, raw_message, customer_email))
        return tid, values

    def submit(self, raw_message: str, customer_email: str | None = None) -> str:
        """Fire-and-forget submission (API). Progress is observable via SSE."""
        tid = new_ticket_id()

        async def _job() -> None:
            await self.run_ticket(raw_message, customer_email, ticket_id=tid)

        task = asyncio.create_task(_job(), name=f"ticket-{tid}")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return tid

    async def resume(self, ticket_id: str, decision: HumanDecision) -> dict[str, Any]:
        async with self._lock(ticket_id):
            snap = await self.graph.aget_state(self._config(ticket_id))
            if not snap.values:
                raise TicketNotFoundError(ticket_id)
            if tuple(snap.next) != ("human_gate",):
                raise InvalidTransitionError(f"ticket {ticket_id} is not awaiting approval (next={list(snap.next)})")
            await self._emit(ticket_id, "decision", "human_gate", decision.model_dump(mode="json"))
            return await self._drive(ticket_id, Command(resume=decision.model_dump(mode="json")))

    def resume_in_background(self, ticket_id: str, decision: HumanDecision) -> None:
        async def _job() -> None:
            await self.resume(ticket_id, decision)

        task = asyncio.create_task(_job(), name=f"resume-{ticket_id}")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def recover_inflight(self) -> list[str]:
        """Crash recovery: resume runs that were interrupted mid-flight (not those awaiting a human)."""
        recovered: list[str] = []
        for row in await self.db.list_tickets(limit=500):
            status = str(row["execution_status"])
            if status in TERMINAL_STATUSES or status == "AWAITING_APPROVAL":
                continue
            tid = str(row["ticket_id"])
            snap = await self.graph.aget_state(self._config(tid))
            if snap.values and snap.next and tuple(snap.next) != ("human_gate",):
                recovered.append(tid)

                async def _job(t: str = tid) -> None:
                    async with self._lock(t):
                        await self._drive(t, None)

                task = asyncio.create_task(_job(), name=f"recover-{tid}")
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)
        return recovered

    # ------------------------------------------------------------------- reads
    async def get_view(self, ticket_id: str) -> TicketView:
        snap = await self.graph.aget_state(self._config(ticket_id))
        row = await self.db.get_ticket_row(ticket_id)
        if not snap.values:
            if row is None:
                raise TicketNotFoundError(ticket_id)
            return TicketView(ticket_id=ticket_id, raw_message=str(row["raw_message"]), execution_status=str(row["execution_status"]))
        pending: dict[str, object] | None = None
        for task in snap.tasks:
            for itr in task.interrupts:
                if isinstance(itr.value, dict):
                    pending = itr.value
        values = dict(snap.values)
        return TicketView.model_validate(
            {
                **values,
                "next_nodes": list(snap.next),
                "pending_interrupt": pending,
                "created_at": row["created_at"] if row else None,
                "updated_at": row["updated_at"] if row else None,
            }
        )

    async def pending_approvals(self) -> list[TicketView]:
        out: list[TicketView] = []
        for row in await self.db.list_tickets(limit=200):
            if row["execution_status"] == "AWAITING_APPROVAL":
                view = await self.get_view(str(row["ticket_id"]))
                if view.pending_interrupt is not None:
                    out.append(view)
        return out
