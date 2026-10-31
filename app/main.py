"""FastAPI entrypoint: REST API, SSE event stream and the operator cockpit."""

from __future__ import annotations

import asyncio
import hmac
import io
import json
import logging
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, ValidationError

from app import __version__
from app.agent.graph import EDGES, NODES
from app.config import PROJECT_ROOT, get_settings
from app.db.seed import seed_database
from app.integrations.email_inbound import InboundEmailPayload, from_webhook_payload
from app.integrations.imap_listener import EmailListener
from app.models.schemas import DecisionRequest, HumanDecision, TicketCreate, TicketCreated, TicketView
from app.scenarios import SCENARIOS
from app.services.inbound_email import InboundEmailService
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
    inbound = InboundEmailService(runtime)
    listener = EmailListener(settings, inbound.ingest)
    app.state.runtime = runtime
    app.state.inbound = inbound
    app.state.listener = listener
    if settings.enable_email_listener:
        listener.start()
        logger.info("IMAP listener started (%s, folder=%s)", settings.imap_host, settings.imap_folder)
    recovered = await runtime.recover_inflight()
    logger.info("SentinelOps ready · llm=%s · recovered=%s", runtime.llm_status(), recovered)
    try:
        yield
    finally:
        await listener.stop()
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


def get_inbound(request: Request) -> InboundEmailService:
    service: InboundEmailService = request.app.state.inbound
    return service


def get_listener(request: Request) -> EmailListener:
    listener: EmailListener = request.app.state.listener
    return listener


InboundDep = Annotated[InboundEmailService, Depends(get_inbound)]
ListenerDep = Annotated[EmailListener, Depends(get_listener)]


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
        "email": {
            "listener_enabled": s.enable_email_listener,
            "sender_enabled": rt.deps.mailer.real_delivery_enabled,
            "support_address": s.support_email_address,
        },
    }


# ------------------------------------------------------------------ email channel
async def _ingest_payload(data: object, inbound: InboundEmailService, *, simulator: bool) -> dict[str, object]:
    if not isinstance(data, dict):
        raise HTTPException(422, "payload must be an object")
    try:
        payload = InboundEmailPayload.model_validate(data)
        if simulator:
            payload = payload.model_copy(update={"source": "simulator"})
        email = from_webhook_payload(payload)
        result = await inbound.ingest(email)
    except ValidationError as exc:
        raise HTTPException(422, exc.errors(include_url=False, include_context=False)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {
        "ticket_id": result.ticket_id,
        "duplicate": result.duplicate,
        "message_id": result.message_id,
        "sender_is_customer": result.sender_is_customer,
    }


@app.post("/api/webhooks/inbound-email", status_code=202)
async def inbound_email_webhook(
    request: Request,
    rt: RuntimeDep,
    inbound: InboundDep,
    x_webhook_token: Annotated[str | None, Header()] = None,
) -> dict[str, object]:
    """Inbound-parse webhook (generic JSON, Mailgun, SendGrid, Postmark; JSON or form-encoded)."""
    secret = rt.settings.inbound_webhook_secret
    if secret is not None and secret.get_secret_value():
        if not hmac.compare_digest((x_webhook_token or "").encode(), secret.get_secret_value().encode()):
            raise HTTPException(401, "invalid or missing X-Webhook-Token")
    content_type = request.headers.get("content-type", "")
    data: object
    if "application/json" in content_type:
        try:
            data = await request.json()
        except json.JSONDecodeError as exc:
            raise HTTPException(400, "malformed JSON") from exc
    else:
        form = await request.form()
        data = {k: v for k, v in form.items() if isinstance(v, str)}
    return await _ingest_payload(data, inbound, simulator=False)


@app.post("/api/email/simulate", status_code=202)
async def simulate_inbound_email(request: Request, inbound: InboundDep) -> dict[str, object]:
    """Cockpit "Send Live Test Email": same pipeline as the webhook, tagged as a simulation.

    Simulated mail is never delivered over SMTP (see Mailer), so the demo cannot email real people.
    """
    try:
        data = await request.json()
    except json.JSONDecodeError as exc:
        raise HTTPException(400, "malformed JSON") from exc
    return await _ingest_payload(data, inbound, simulator=True)


@app.get("/api/email/status")
async def email_status(rt: RuntimeDep, listener: ListenerDep) -> dict[str, object]:
    s = rt.settings
    return {
        "support_address": s.support_email_address,
        "supervisor_email": s.supervisor_email,
        "webhook": {"path": "/api/webhooks/inbound-email", "token_required": bool(s.inbound_webhook_secret)},
        "listener": listener.describe(),
        "sender": {
            "enabled": rt.deps.mailer.real_delivery_enabled,
            "host": s.smtp_host if s.enable_email_sender else None,
        },
    }


@app.post("/api/email/poll")
async def email_poll_now(listener: ListenerDep) -> dict[str, object]:
    """Trigger one IMAP poll immediately (useful in demos instead of waiting for the interval)."""
    if not listener.status.enabled:
        raise HTTPException(409, "IMAP listener disabled (set ENABLE_EMAIL_LISTENER=true)")
    created = await listener.poll_once()
    return {"created": created, "status": listener.describe()}


@app.get("/api/inbox")
async def inbox(rt: RuntimeDep, limit: Annotated[int, Query(ge=1, le=500)] = 100) -> list[dict[str, object]]:
    return await rt.db.list_inbound_emails(limit)


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
        "alerts": await rt.db.list_supervisor_alerts(),
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
