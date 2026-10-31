# 🏛️ SentinelOps

**Enterprise Autonomous Financial Dispute & Agentic Resolution Engine**: a multi-step LangGraph agent,
deterministic and resilient, with self-correction, Human-in-the-Loop (HITL), LLM-as-a-Judge and an automated
evaluation harness for CI/CD.

Created by **Santino Genta**.

```
[Ticket] → triage → db_inspector ⟲ (self-correction ≤3) → policy_retriever → resolution_planner
                          ▲                                                    │  │
                          └──────────── re-query (wider window) ───────────────┘  │
             ┌─────────────────────── > $200 / high risk / injection ─────────────┤
             ▼                                                                    ▼ ≤ $200 & low risk
        human_gate (interrupt) ── approved ──► action_executor (Mock Stripe) ◄────┘
             └── rejected ───────────────────► response_drafter ◄── llm_judge (score < 0.85 → re-draft)
                                                                        └── ≥ 0.85 → dispatch → END
```

## Quick start (Windows / PowerShell)

```powershell
cd sentinelops
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

uvicorn app.main:app --reload --port 8000     # cockpit at http://127.0.0.1:8000
pytest tests/eval_suite.py -v                  # evaluation suite (CI gate)
pytest                                         # everything: unit + integration + API + email (125 tests)
mypy                                           # strict typing (--strict)
```

In bash: `export GEMINI_API_KEY=...`; in PowerShell: `$env:GEMINI_API_KEY="..."`. It is also read from `.env`.

### Configuration (`.env`, see `.env.example`)

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `mock` | `gemini` · `groq` · `mock` (deterministic engine, no network) |
| `GEMINI_API_KEY` / `GEMINI_MODEL` | — / `gemini-2.5-flash` | Google AI Studio (free tier) |
| `GROQ_API_KEY` / `GROQ_MODEL` | — / `llama-3.3-70b-versatile` | OpenAI-compatible alternative |
| `LLM_RPM` | `10` | Client-side rate limiter (free tier) |
| `EMBEDDING_PROVIDER` | `local` | `local` (deterministic hashing) · `gemini` (`gemini-embedding-001`) |
| `HITL_THRESHOLD_USD` | `200` | Human approval threshold |
| `JUDGE_PASS_THRESHOLD` | `0.85` | Minimum faithfulness score |
| `MAX_TOOL_RETRIES` / `MAX_REDRAFTS` | `3` / `2` | Self-correction budgets |
| `AUTOPILOT_DEFAULT` / `AUTOPILOT_MAX_USD` | `true` / `1000` | Initial Auto-Pilot state and automatic-approval ceiling |
| `STRIPE_API_KEY` / `STRIPE_MIRROR_REFUNDS` | — / `true` | `sk_test_…` key for the Stripe dashboard and the refund mirror |
| `EMAIL_LOOKBACK_MINUTES` / `EMAIL_SUBJECT_KEYWORDS` | `120` / empty | How far back the mailbox is read · optional subject filter |

## Design principles

1. **The LLM understands language; code decides money.** Payout amounts **always** come from the verified
   ledger (`app/agent/engine.py`), never from the customer's text or the model's free-form output. This makes it
   impossible by construction for a hallucination or a prompt injection to inflate a payout.
2. **No raw dicts at the boundaries.** Every LLM output, tool argument, DB row, API payload and human decision
   goes through strict Pydantic v2 models (`extra="forbid"`, regexes, cross-field validators). The LangGraph state
   stores the JSON form (`model_dump`) so the SQLite checkpointer can persist it, and each node re-hydrates it with
   validation.
3. **Graceful degradation.** Every LLM node has a deterministic path. A circuit breaker opens on repeated failures
   and **permanently** on authentication errors (401/403); the agent keeps resolving with the deterministic engine
   instead of failing.
4. **Defense in depth.** The HITL gate lives in the routing **and** the `action_executor` re-checks the approval
   before moving money. The Mock Stripe enforces idempotency keys and never refunds more than the balance.

## Architecture

| Layer | File | What it does |
|---|---|---|
| State | `app/models/state.py` | `AgentState` (SPEC §4.1 + extensions), append-only reducers for `logs`/`tool_calls` |
| Models | `app/models/schemas.py` | Strict Pydantic v2 models for the whole domain |
| Graph | `app/agent/graph.py` | `StateGraph`, conditional edges, self-correction/re-draft loops |
| Nodes | `app/agent/nodes.py` | 9 nodes; `human_gate` uses `interrupt()` and resumes with `Command(resume=...)` |
| Engine | `app/agent/engine.py` | Duplicate detection, gateway sync errors, fraud signals, per-tier windows, risk, plan |
| Guardrails | `app/agent/guardrails.py` | Prompt-injection detection, fencing of untrusted input, grounding check |
| LLM | `app/llm/` | Gemini (`google-genai`) / Groq; JSON mode + validation + self-repair, rate limiter, retries with backoff, circuit breaker |
| RAG | `app/rag/` | `policies.md` (12 clauses) → ChromaDB (dense) + BM25 (sparse) fused with RRF |
| Integrations | `app/integrations/` | Mock Stripe (idempotent, transactional), Mock CRM + email outbox, fault injector |
| Runtime | `app/services/runtime.py` | `AsyncSqliteSaver`, background execution, SSE bus, HITL resume, crash recovery |
| API | `app/main.py` | Async FastAPI + SSE |
| Cockpit | `app/static/index.html` | SPA (Tailwind + Lucide): live graph, timeline, HITL modal, ledger, evals, chaos |

### Key behaviors

- **Tool self-correction (DBInspector ⟲):** invalid arguments (e.g. `TXN-HENRY000`) → validation error
  → repair (LLM or deterministic → `txn_henry000`) → retry. Timeouts/503 → retry. Up to 3 retries;
  then it escalates to a human **without moving money** (POL-ESC-001).
- **Planner → DBInspector loop:** if the customer cites a charge outside the queried window (90 days), the
  planner asks for a re-query over 365 days.
- **HITL:** payout > $200, payout with HIGH risk (fraud, ≥3 chargebacks, account < 7 days old) or any
  manipulation attempt → `interrupt()`. The state survives process restarts (SQLite checkpoint; covered by a test).
- **LLM-as-a-Judge:** the final score is `min(LLM_score, deterministic_grounding_score)`, so a "fooled" judge
  cannot let an invented amount through. < 0.85 → re-draft with feedback; once the budget is spent, a template
  anchored to the facts is used.
- **Anti-hallucination in triage:** email/amount/references extracted by the LLM are discarded if they do not
  appear verbatim in the message.

## 📧 Real email channel (inbound mailbox + worker)

Customers can write by email and the agent resolves the case and answers in the same thread.

```
Gmail / Outlook ──IMAP──► EmailListener ─┐
Mailgun / SendGrid / Postmark ─webhook──►├─► InboundEmailService ─► LangGraph ─► HTML reply "Re: <subject>"
Cockpit "📧 Send Live Test Email" ───────┘   (Message-Id dedupe,      │            (outbox + optional SMTP)
                                              sender identity)        └─ HumanGate ─► ⚠️ supervisor alert
```

| Piece | Where | Details |
|---|---|---|
| Webhook | `POST /api/webhooks/inbound-email` | JSON or form-encoded. Accepts `from/subject/body_plain/message_id` and the Mailgun (`sender`, `body-plain`, `Message-Id`), SendGrid (`text`) and Postmark (`From`, `TextBody`, `MessageID`) aliases. With `INBOUND_WEBHOOK_SECRET` it requires the `X-Webhook-Token` header. |
| IMAP listener | `app/integrations/imap_listener.py` | `ENABLE_EMAIL_LISTENER=true`. Reads the mail that arrived in the last `EMAIL_LOOKBACK_MINUTES` (120 by default), **read or unread** (an email already opened on the phone is still processed), newest first. Any subject works; the `EMAIL_SUBJECT_KEYWORDS` filter is optional. It skips bounces, auto-replies, newsletters, no-reply senders and the replies the agent itself sent. Each UID is evaluated once and the `Message-Id` prevents duplicate tickets after a restart. It marks `\Seen` **after** creating the ticket and backs off exponentially on errors. The **📥 Revisar Bandeja Ahora** button (`POST /api/email/poll-now`) forces an immediate read and reports what it found. |
| Simulator | **📧 Send Live Test Email** button | Gmail-style window with templates. Uses the same pipeline (`POST /api/email/simulate`) and the graph starts on its own via SSE. |
| Reply | **Ledger / Outbox** tab | Preview of the rendered HTML email (sandboxed iframe), with `Re: …` subject and `In-Reply-To`. |
| Supervisor | **⚠️ New Email requires billing supervisor approval** banner | Fires when the run enters `HumanGate`. It also stores (and optionally sends) an alert email and updates the tab title. If the browser allows it, it shows a desktop notification. |

**Safety guarantees**
- **Idempotency:** the same `Message-Id` (provider retries, IMAP re-reads) produces a single ticket.
- **Identity:** if the sender owns the account, that is the authenticated identity. If they write from another address (e.g. a personal email that mentions `alice@acme-corp.com`), risk goes up to HIGH (`sender_not_account_owner`). In that case any payout requires human approval and the reply does not reveal the account holder's name (POL-ID-001).
- **Real SMTP only for real mail:** cockpit simulations never go out over SMTP, so the demo can never write to real domains.
- **No mail loops:** every agent reply carries `Auto-Submitted: auto-replied`, the `X-SentinelOps-Ticket` header and its own `Message-ID`, and the listener never re-processes it. That is why you can write from the support mailbox to itself (self-send in demo mode).
- **Never writes to strangers:** the SMTP reply only goes to the account owner, a verified alias or the mailbox itself. Everything else stays in the outbox.

**Try it with real Gmail:** turn on 2FA, create an *App Password* and either connect the mailbox from the cockpit
(**⚙️ Conectar Casilla de Correo**) or set in `.env`: `ENABLE_EMAIL_LISTENER=true`, `IMAP_USER=you@gmail.com`,
`IMAP_PASSWORD=<app password>` and, to reply, `ENABLE_EMAIL_SENDER=true`. Send an email from a seed customer
account (or a verified alias, or mentioning one in the body) and watch the cockpit pick it up on its own.

## ⚡ Auto-Pilot, mailbox connected from the UI and live Stripe

**Auto-Pilot (zero-touch).** The **⚡ Auto-Pilot: ON/OFF** toggle in the header (persisted in the database) does not
skip the `HumanGate`: the graph still pauses with `interrupt()` (so it stays audited). What changes is who resumes it:
if the case complies with policy, the runtime resumes it with the `auto-pilot` decision. It executes the refund,
mirrors it to Stripe, passes the LLM Judge and dispatches the reply in ~4 s with zero clicks. **It still waits for a
human** when there is: prompt injection · a sender who does not own the account · HIGH risk (fraud, chargebacks, new
account) · an amount above `AUTOPILOT_MAX_USD` · any action other than a simple refund. In those cases the
`autopilot_blocked` event (with the reasons) and the supervisor alert are emitted.

**⚙️ Conectar Casilla de Correo (connect mailbox).** A modal with Gmail / Google Workspace, Outlook / Office 365 or
custom IMAP, and a step-by-step guide to create the App Password. **🧪 Probar Conexión** (`POST /api/email/test-connection`)
performs a real IMAP (and SMTP) login without saving anything. **💾 Guardar y Activar** (`POST /api/email/save-config`)
re-checks, saves to `data/sentinelops.db` (outside git), applies the config hot and restarts the listener without
restarting uvicorn. The listener reuses a single IMAP session (re-`SELECT` on every poll), so Gmail does not throttle it
even when listening every 5 s. The header shows `🟢 Correo Activo: … (Escuchando en vivo cada 5s)` or
`⚪ Correo no conectado`. The App Password is never returned by the API. *Demo mode:* you can optionally link your
personal email to a ledger customer (verified alias) to try the full real loop: you write from Gmail, the refund is
made in Stripe and the reply lands in your inbox.

**Live follow in the cockpit.** When an email arrives, the cockpit jumps to the new ticket on the **Ticket** tab,
shows the received email, animates the LangGraph nodes in real time and, when the refund finishes, shows the
Stripe receipt on screen (refund ID, PaymentIntent, `succeeded` status, **↗ Abrir en Stripe.com**).

**💳 Stripe Live Dashboard.** `GET /api/stripe/dashboard-sync` uses stripe-python to fetch the real balance, refunds
and charges. Every ledger refund is **mirrored into Stripe test mode**: the original charge is created once
(test card `pm_card_bypassPending`) and then the `Refund`, with idempotency key = local refund id. If Stripe fails,
the ticket does not fail: the mirror is recorded as `failed` and stays visible. **With an `sk_live_` key the dashboard
is read-only**: SentinelOps never creates real charges or refunds. Each row has **↗ Abrir en Stripe.com**.
Note: the test balance can go negative, because Stripe does not return its fee on refunds.

## API

| Method | Route | Description |
|---|---|---|
| `POST` | `/api/tickets` | Creates a ticket (202) and runs the agent in the background |
| `GET` | `/api/tickets` · `/api/tickets/{id}` · `/api/tickets/{id}/events` | List · full checkpoint state · event history |
| `POST` | `/api/tickets/{id}/decision` | HITL decision `{approved, reviewer, reason}` (409 if not waiting) |
| `GET` | `/api/approvals` | Tickets paused at `human_gate` |
| `GET` | `/api/stream` | Server-Sent Events (`node_started`, `node_completed`, `interrupt`, `decision`, `run_completed`, …) |
| `GET` | `/api/ledger` · `/api/graph` · `/api/health` · `/api/eval/report` | Refunds/outbox · topology · LLM status · latest eval report |
| `POST` | `/api/admin/faults` · `/api/admin/reset` | Fault injection (chaos) · ledger re-seed |
| `POST` | `/api/webhooks/inbound-email` · `/api/email/simulate` | Inbound email (provider, with token) · cockpit simulator |
| `GET`/`POST` | `/api/email/status` · `/api/inbox` · `/api/email/poll-now` | Channel status · received emails · immediate IMAP read |
| `GET`/`POST` | `/api/email/config` · `/api/email/test-connection` · `/api/email/save-config` · `/api/email/disconnect` | Mailbox connected from the UI (password never exposed) |
| `GET`/`PUT` | `/api/autopilot` | Auto-Pilot state `{enabled, max_usd}` |
| `GET` | `/api/stripe/dashboard-sync` | Real-time Stripe balance + refunds + charges |

Interactive OpenAPI docs: `http://127.0.0.1:8000/docs`.

## Evaluation suite (`tests/eval_suite.py`)

18 golden cases (`tests/datasets/golden_tickets.json`): duplicates, fraud, gateway error, refund windows,
already refunded, unknown customer, Spanish, 4 prompt-injection attacks and 2 chaos scenarios.

| Metric | Threshold | Result (deterministic) |
|---|---|---|
| Tool Precision / Recall | ≥ 0.95 | 1.00 / 1.00 |
| Hallucination Rate | ≤ 0.00 | 0.00 |
| Prompt Injection Defense | = 1.00 | 1.00 |
| HITL Gate Accuracy | = 1.00 | 1.00 |
| Action Accuracy | ≥ 0.95 | 1.00 |

The metrics are implemented natively following DeepEval semantics (ToolCorrectness, Faithfulness/Hallucination),
without depending on an OpenAI LLM to run in CI. `EVAL_LIVE=1` runs the same set against the configured provider.
The JSON report is written to `reports/eval_report.json` and shown in the cockpit's **Evals** tab.

## Demo data

Seed customers (`app/db/seed.py`, dates relative to the seed time): `alice@acme-corp.com` (duplicate $149),
`bob@startup.io` (duplicate $480), `carol@gmail.com` (fraud $899.99 from RU), `dave@example.com` (gateway
timeout $59), `erin@designhub.co` ($29 within window), `frank@oldmail.net` (outside window), `grace@lumen.io`
(already refunded), `mallory@protonmail.com` (new account, 3 chargebacks), `henry@bigretail.com` ($1,250 and a
charge from 200 days ago). The cockpit has one button per scenario.

## Author

**Santino Genta** — design and development of SentinelOps.
