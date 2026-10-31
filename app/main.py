"""FastAPI entrypoint: REST API, SSE event stream and the operator cockpit."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from app import __version__
from app.agent.graph import EDGES, NODES
from app.config import PROJECT_ROOT, get_settings
from app.db.seed import seed_database
from app.models.schemas import DecisionRequest, HumanDecision, TicketCreate, TicketCreated, TicketView
from app.scenarios import SCENARIOS
from app.services.runtime import InvalidTransitionError, SentinelRuntime, TicketNotFoundError

for _stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
    if isinstance(_stream, io.TextIOWrapper):
        _stream.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("sentinelops.api")

STATIC_DIR = Path(__file__).parent / "static"
REPORT_PATH = PROJECT_ROOT / "reports" / "eval_report.json"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    runtime = await SentinelRuntime.create(settings)
    app.state.runtime = runtime
    recovered = await runtime.recover_inflight()
    logger.info("SentinelOps ready · llm=%s · recovered=%s", runtime.llm_status(), recovered)
    try:
        yield
    finally:
        await runtime.close()


app = FastAPI(
    title="SentinelOps",
    version=__version__,
    description="Enterprise Autonomous Financial Dispute & Agentic Resolution Engine",
    lifespan=lifespan,
)


def get_runtime(request: Request) -> SentinelRuntime:
    runtime: SentinelRuntime = request.app.state.runtime
    return runtime


RuntimeDep = Annotated[SentinelRuntime, Depends(get_runtime)]


class FaultRequest(BaseModel):
    tool: Literal["db.query_transactions", "stripe.create_refund", "email.send"]
    times: int = Field(default=1, ge=1, le=10)
    kind: Literal["timeout", "error", "invalid_args"] = "timeout"


# --------------------------------------------------------------------------- UI
@app.get("/", include_in_schema=False)
async def cockpit() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


# ----------------------------------------------------------------------- system
@app.get("/api/health")
async def health(rt: RuntimeDep) -> dict[str, object]:
    s = rt.settings
    return {
        "status": "ok",
        "version": __version__,
        "llm": rt.llm_status(),
        "embedder": rt.deps.retriever.embedder.name,
        "hitl_threshold_usd": s.hitl_threshold_usd,
        "judge_pass_threshold": s.judge_pass_threshold,
        "max_tool_retries": s.max_tool_retries,
        "max_redrafts": s.max_redrafts,
    }


@app.get("/api/graph")
async def graph_topology() -> dict[str, object]:
    return {"nodes": list(NODES), "edges": [{"source": a, "target": b, "label": c} for a, b, c in EDGES]}


@app.get("/api/scenarios")
async def scenarios() -> list[dict[str, object]]:
    return [dict(s) for s in SCENARIOS]


# ---------------------------------------------------------------------- tickets
@app.post("/api/tickets", status_code=202, response_model=TicketCreated)
async def create_ticket(body: TicketCreate, rt: RuntimeDep) -> TicketCreated:
    tid = rt.submit(body.message, body.customer_email)
    return TicketCreated(ticket_id=tid, execution_status="TRIAGING")


@app.get("/api/tickets")
async def list_tickets(rt: RuntimeDep, limit: Annotated[int, Query(ge=1, le=500)] = 100) -> list[dict[str, object]]:
    return await rt.db.list_tickets(limit)


@app.get("/api/tickets/{ticket_id}", response_model=TicketView)
async def get_ticket(ticket_id: str, rt: RuntimeDep) -> TicketView:
    try:
        return await rt.get_view(ticket_id)
    except TicketNotFoundError as exc:
        raise HTTPException(404, f"ticket {ticket_id} not found") from exc


@app.get("/api/tickets/{ticket_id}/events")
async def ticket_events(ticket_id: str, rt: RuntimeDep) -> list[dict[str, object]]:
    return await rt.db.list_events(ticket_id)


@app.post("/api/tickets/{ticket_id}/decision", status_code=202)
async def decide(ticket_id: str, body: DecisionRequest, rt: RuntimeDep) -> dict[str, object]:
    try:
        view = await rt.get_view(ticket_id)
    except TicketNotFoundError as exc:
        raise HTTPException(404, f"ticket {ticket_id} not found") from exc
    if view.next_nodes != ["human_gate"]:
        raise HTTPException(409, f"ticket {ticket_id} is not awaiting approval")
    decision = HumanDecision.model_validate(body.model_dump())
    rt.resume_in_background(ticket_id, decision)
    return {"ticket_id": ticket_id, "accepted": True, "approved": decision.approved}


@app.get("/api/approvals", response_model=list[TicketView])
async def approvals(rt: RuntimeDep) -> list[TicketView]:
    return await rt.pending_approvals()


# ----------------------------------------------------------------------- ledger
@app.get("/api/ledger")
async def ledger(rt: RuntimeDep) -> dict[str, object]:
    return {
        "refunds": await rt.db.list_refunds(),
        "outbox": await rt.db.list_outbox(),
        "customers": await rt.db.list_customers(),
    }


# ------------------------------------------------------------------------ admin
@app.post("/api/admin/faults")
async def inject_fault(body: FaultRequest, rt: RuntimeDep) -> dict[str, object]:
    rt.deps.faults.fail_next(body.tool, body.times, body.kind)
    return {"armed": body.model_dump()}


@app.post("/api/admin/reset")
async def reset(rt: RuntimeDep) -> dict[str, object]:
    rt.deps.faults.clear()
    await seed_database(rt.db, force=True)
    return {"reset": True}


@app.get("/api/eval/report")
async def eval_report() -> JSONResponse:
    if not REPORT_PATH.exists():
        return JSONResponse({"available": False, "hint": "run: pytest tests/eval_suite.py -v"})
    return JSONResponse({"available": True, **json.loads(REPORT_PATH.read_text(encoding="utf-8"))})


# -------------------------------------------------------------------------- SSE
@app.get("/api/stream")
async def stream(request: Request, rt: RuntimeDep, ticket_id: str | None = None) -> StreamingResponse:
    async def gen() -> AsyncIterator[str]:
        yield "retry: 2000\n\n"
        sub = rt.bus.subscribe()
        nxt: asyncio.Task[dict[str, object]] | None = None
        try:
            while not await request.is_disconnected():
                if nxt is None:
                    nxt = asyncio.ensure_future(anext(sub))
                done, _ = await asyncio.wait({nxt}, timeout=15)
                if not done:
                    yield ": keep-alive\n\n"
                    continue
                event = nxt.result()
                nxt = None
                if ticket_id and event.get("ticket_id") != ticket_id:
                    continue
                yield f"event: {event['type']}\ndata: {json.dumps(event, default=str)}\n\n"
        finally:
            if nxt is not None:
                nxt.cancel()
            await sub.aclose()

    return StreamingResponse(
        gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )
