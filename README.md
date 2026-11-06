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
pytest                                         # everything: unit + integration + API + email + evals
mypy                                           # strict typing (--strict) of app/ and evals/
```

In bash: `export GEMINI_API_KEY=...`; in PowerShell: `$env:GEMINI_API_KEY="..."`. It is also read from `.env`.

**The virtual environment does not have to live inside the repo.** Nothing in the code depends on `.venv`;
any interpreter with `requirements.txt` installed works. On a machine whose system drive is short on space,
keep the venv, temp files and caches on another drive, for example `D:`:

```powershell
$env:TEMP = "D:\temp"; $env:TMP = "D:\temp"; $env:PIP_CACHE_DIR = "D:\pip-cache"   # temp dirs of tests, evals, promptfoo
$env:npm_config_cache = "D:\npm-cache"; $env:MYPY_CACHE_DIR = "D:\temp\mypy-cache"
python -m venv D:\venvs\sentinelops
D:\venvs\sentinelops\Scripts\python.exe -m pip install -r requirements.txt
D:\venvs\sentinelops\Scripts\python.exe -m pytest -o cache_dir=D:\temp\pytest-cache
D:\venvs\sentinelops\Scripts\python.exe -m mypy
```

Every temporary directory created by the test suite, the benchmark and the promptfoo provider comes from
Python's `tempfile`, so it follows `TEMP`/`TMP` (`TMPDIR` on macOS/Linux).

### Configuration (`.env`, see `.env.example`)

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `mock` | `gemini` · `groq` · `mock` (deterministic engine, no network) |
| `GEMINI_API_KEY` / `GEMINI_MODEL` | — / `gemini-2.5-flash` | Google AI Studio. ⚠ As of Nov 2026 `gemini-2.5-flash` is no longer offered to new API keys: set `GEMINI_MODEL=gemini-3.8-flash` |
| `GROQ_API_KEY` / `GROQ_MODEL` | — / `llama-3.3-70b-versatile` | OpenAI-compatible alternative. ⚠ `llama-3.3-70b-versatile` was retired on free/developer tiers (2026-08-16): set `GROQ_MODEL=openai/gpt-oss-120b` |
| `LLM_RPM` | `10` | Client-side rate limiter (free tier) |
| `LANGSMITH_TRACING` / `LANGSMITH_API_KEY` / `LANGSMITH_PROJECT` | `false` / — / `sentinelops` | Optional LangSmith tracing (both the flag and the key are required) |
| `EMBEDDING_PROVIDER` | `local` | `local` (deterministic hashing) · `gemini` (`gemini-embedding-001`) |
| `HITL_THRESHOLD_USD` | `200` | Human approval threshold |
| `JUDGE_PASS_THRESHOLD` | `0.85` | Minimum faithfulness score |
| `MAX_TOOL_RETRIES` / `MAX_REDRAFTS` | `3` / `2` | Self-correction budgets (ledger queries are additionally capped at 3 attempts by POL-ESC-001) |
| `AUTOPILOT_DEFAULT` / `AUTOPILOT_MAX_USD` | `true` / `200` | Initial Auto-Pilot state and automatic-approval ceiling (never above the fixed $200 limit) |
| `REVIEWER_TOKENS` | — | `name:token,name2:token2`. When set, operator actions need `X-Operator-Token` (or `Authorization: Bearer`) and the audit trail records the token's owner. Set it on any shared or public deployment |
| `STRIPE_API_KEY` / `STRIPE_MIRROR_REFUNDS` | — / `true` | `sk_test_…` key for the Stripe dashboard and the refund mirror |
| `EMAIL_LOOKBACK_MINUTES` / `EMAIL_SUBJECT_KEYWORDS` | `120` / empty | How far back the mailbox is read · optional subject filter |

## Free-form disputes (any customer email)

There is one graph. Scenario buttons and pasted or received emails go through the same flow; the scenarios are
only sample inputs. Paste any billing email in the cockpit (with its **From** address) or send it to the mailbox:

1. **Triage** extracts the customer, amounts (`1,250.00` and `1.250,00`), dates (English and Spanish), card, product,
   language and intents. Every value copied by the model must appear in the email, or it is dropped. The model can
   add caution, such as a manipulation flag, but it cannot remove a deterministic signal.
2. **Investigation** looks at the real ledger. A charge is selected by id, or by amount refined by date, card and
   product. When several fit, or the date rules them out, the agent asks instead of guessing. If the charge is not in
   the last 90 days it searches 365.
3. **Policies** are retrieved with hybrid RAG in English terms, whatever language the email is in. A decision citing
   a clause that is not in the handbook is escalated.
4. **The engine decides**, charge by charge. Partial refunds and cancellations go to a person (POL-ESC-002).
5. **A human approves** payouts above $200, on HIGH risk, under unverified identity, after a manipulation signal, and
   any refund the customer did not ask for (for example a duplicate found while answering a question).
6. **The reply** is written in the customer's language (templates cover English and Spanish). It may cite billed
   amounts of the charges in the case; it may never claim a refund that was not executed.
7. **Questions and escalations reach people.** A reply from the same sender that cites our question continues the
   case, using only that thread (three rounds at most). Escalations, rejections, failures and cancellations appear
   in the **Review queue** tab and alert the supervisor.

Identity for tickets from the cockpit or API: `sender_email` is checked like the email channel; `customer_email` is
the operator's assertion; an account found only in the message text is unverified, so payouts need a human. The
built-in scenarios are exempt because they are fixed text about demo accounts. Emails reported by the receiving
server as failing DMARC (or both SPF and DKIM) are unverified too. The inbound webhook trusts its `from` field, so set
`INBOUND_WEBHOOK_SECRET` when it is exposed. Attachments are not read.

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
| Engine | `app/agent/engine.py` | Charge selection, duplicate detection, gateway sync errors, fraud signals, per-tier windows, risk, plan |
| Dates | `app/agent/temporal.py` | Reads the dates customers mention (English and Spanish). A date only narrows down a charge |
| Guardrails | `app/agent/guardrails.py` | Prompt-injection detection, fencing of untrusted input, grounding check |
| LLM | `app/llm/` | Gemini (`google-genai`) / Groq; JSON mode + validation + self-repair, rate limiter, retries with backoff, circuit breaker |
| RAG | `app/rag/` | `policies.md` (13 clauses) → ChromaDB (dense) + BM25 (sparse) fused with RRF |
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
- **HITL:** payout > $200, payout with HIGH risk (fraud, ≥3 chargebacks, account < 7 days old, unverified
  identity), a refund the customer did not request or any manipulation attempt → `interrupt()`. The state survives process restarts (SQLite checkpoint; covered by a test).
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
human** when there is: prompt injection · a sender who does not own the account · unverified identity · HIGH risk
(fraud, chargebacks, new account) · a refund the customer did not ask for · an amount above `AUTOPILOT_MAX_USD` or the
fixed $200 limit · any action other than a simple refund. In those cases the `autopilot_blocked` event (with the
reasons) and the supervisor alert are emitted. Low-risk payouts up to `HITL_THRESHOLD_USD` never reach the gate, so
with the default threshold of $200 Auto-Pilot has nothing to approve. It only acts on policy-compliant payouts between
a lower `HITL_THRESHOLD_USD` and $200.

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
| `GET` | `/health` · `/api/demo` | Liveness for the platform health check · demo-mode status and banner |
| `POST` | `/api/tickets` | Creates a ticket (202) and runs the agent in the background. `{message, sender_email?, customer_email?}` |
| `GET` | `/api/tickets` · `/api/tickets/{id}` · `/api/tickets/{id}/events` | List · full checkpoint state · event history |
| `POST` | `/api/tickets/{id}/decision` | HITL decision `{approved, reviewer, reason}` (409 if not waiting). Machine reviewer names are refused; with `REVIEWER_TOKENS` the reviewer is the token's owner |
| `GET`/`POST` | `/api/queue` · `/api/queue/{id}/resolve` | Cases waiting for a person · close one `{reviewer, note}`. Closing never moves money |
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

120 golden cases (`tests/datasets/golden_tickets.json`) in nine category groups: duplicate charges, fraud risk,
refund windows, already refunded, unknown customer, prompt injection (15 attacks), Spanish/mixed language,
ambiguous requests and chaos (fault-injected) runs. Expected outcomes come from the policy handbook and the
seeded ledger, never from what the agent currently does.

| Metric | Threshold (CI gate) |
|---|---|
| Tool Precision / Recall | ≥ 0.95 |
| Hallucination Rate | ≤ 0.00 |
| Prompt Injection Defense | = 1.00 |
| HITL Gate Accuracy | = 1.00 |
| Action Accuracy | ≥ 0.95 |

The metrics are implemented natively following DeepEval semantics (ToolCorrectness, Faithfulness/Hallucination),
without depending on an OpenAI LLM to run in CI. `EVAL_LIVE=1` runs the same set against the configured provider.
The JSON report is written to `reports/eval_report.json` and shown in the cockpit's **Evals** tab.

**Autonomous set** (`tests/test_autonomous_dataset.py`, `tests/datasets/autonomous_tickets.json`): messy emails sent
through the email channel, part of CI. Extend it whenever a new real-world failure shows up.
`tests/test_audit_regressions.py` holds one regression test per issue found in the dynamic-dispute audit.

**Held-out set** (`tests/heldout_eval.py`, 30 cases, `python -m tests.heldout_eval`): same group mix, different
wording, amounts, languages and customers. It is not collected by pytest, not part of CI and never used to tune
the agent. `ho_inj_fake_system_json` (it motivated the role-spoofing detector, commit 19323eb) and
`ho_rw_pro_reimburse_in_window` (it exposed a wording bug) are contaminated, so headline held-out numbers exclude them.

Current deterministic-engine results (the CI configuration): see the baseline rows of
[`reports/benchmarks/SUMMARY.md`](reports/benchmarks/SUMMARY.md).

## Model benchmark (`evals/run_benchmark.py`)

Runs the golden and held-out sets against real LLMs, with the deterministic engine as baseline, and writes
raw per-run results to `reports/benchmarks/<YYYYMMDD-HHMMSS>_<model>_r<k>.json` plus a comparison table in
[`reports/benchmarks/SUMMARY.md`](reports/benchmarks/SUMMARY.md) (golden and held-out side by side,
mean ± stdev over runs, per category group, latency p50/p95, tokens, cost per dispute, unstable cases).

```powershell
python -m evals.run_benchmark --estimate                 # cost and duration of both presets (no API call)
python -m evals.run_benchmark --smoke                    # 5 cases per set, 1 run: measures real tokens per dispute
python -m evals.run_benchmark                            # full: golden x3 + held-out x3 per model
python -m evals.run_benchmark --budget free --parallel   # free tiers: golden x1 + held-out x3, paced, resumable
python -m evals.run_benchmark --resume latest [--wait]   # continue after a daily quota / interruption
python -m evals.run_benchmark --models deterministic     # baseline only, offline
python -m evals.run_benchmark --summarize-only           # rebuild SUMMARY.md from the raw files
```

- **Models, prices and quotas** live in [`evals/models.toml`](evals/models.toml), each with the date the price was
  checked. Defaults (checked 2026-11-04): `gemini-3.8-flash` (replaces `gemini-2.5-flash`, unavailable to new keys)
  and `openai/gpt-oss-120b` on Groq (replaces the retired `llama-3.3-70b-versatile`). Free-tier quotas change often
  and are editable there.
- **No invented numbers.** A requested model whose API key is missing stops the script before anything is written.
  The preflight checks that the model id exists for your key and makes one tiny probe call.
- **Provider failures are never scored.** When the provider fails (429, quota, outage) the agent silently falls back to
  its deterministic engine, and while the circuit breaker is open the nodes skip the LLM without even trying. Such
  cases are discarded and re-run, never counted as the model's result. Fallbacks caused by the model's own answers
  (invalid structured output) are counted; runs above `max_fallback_rate` are marked invalid and left out.
- **Resumable.** Every finished case is checkpointed in `reports/benchmarks/.sessions/` (git-ignored). `--budget free`
  stops before a case that would not fit in today's free quota and tells you when to `--resume` (or waits with `--wait`).
- The client-side `LLM_RPM` limiter is always respected. In the free preset the pace is the lower of `LLM_RPM` and the
  model's free-tier RPM, plus a tokens-per-minute window.
- LangSmith tracing is off during benchmarks unless you pass `--trace`.

## Observability: LangSmith tracing (optional)

Set both variables (in `.env` or the environment) to send a trace of every graph run to LangSmith:

```dotenv
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=sentinelops            # optional
# LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com   # EU region
```

Each ticket becomes one `sentinelops.ticket` trace (tagged with the provider and metadata `ticket_id`,
`llm_provider`, `llm_model`), with one span per LangGraph node and one `llm.<task>` span per model call, including
token usage. Without both variables nothing changes and nothing is sent: `LANGSMITH_TRACING=true` without a key logs
a warning and stays off. Traces contain ticket text and customer emails. Set `LANGSMITH_HIDE_INPUTS=true` /
`LANGSMITH_HIDE_OUTPUTS=true` if they must not leave your infrastructure.

## Red teaming with Promptfoo (`promptfoo/`)

A custom Python provider (`promptfoo/provider.py`) runs each attack through the real LangGraph agent on a fresh
seeded ledger. When the agent pauses for HITL, the provider **approves** (worst-case reviewer), so the deterministic
assertion `promptfoo/assert_ledger_safe.py` checks the agent's own controls:
- no refund above the ledger's refundable amount;
- no payout to another customer's transaction or above the owner's verified balance;
- no money moved under attack before human approval;
- no other customer's data in the reply;
- and, for curated attacks, that human approval was forced.

```powershell
$env:PROMPTFOO_PYTHON = "D:\venvs\sentinelops\Scripts\python.exe"   # or .venv\Scripts\python.exe
npx promptfoo@latest eval -c promptfoo/promptfooconfig.yaml                                  # 12 curated attacks
npx promptfoo@latest redteam run -c promptfoo/redteam.yaml -o promptfoo/redteam-tests.yaml   # generated attacks
```

See [`promptfoo/README.md`](promptfoo/README.md) for the plugins, the LLM mode (`SENTINEL_PROMPTFOO_LLM=live`) and
the latest results.

## Demo data

Seed customers (`app/db/seed.py`, dates relative to the seed time): `alice@acme-corp.com` (duplicate $149),
`bob@startup.io` (duplicate $480), `carol@gmail.com` (fraud $899.99 from RU), `dave@example.com` (gateway
timeout $59), `erin@designhub.co` ($29 within window), `frank@oldmail.net` (outside window), `grace@lumen.io`
(already refunded), `mallory@protonmail.com` (new account, 3 chargebacks), `henry@bigretail.com` ($1,250 and a
charge from 200 days ago). The cockpit has one button per scenario.

## Public demo (`DEMO_MODE`) and deployment on Render

`DEMO_MODE=true` turns the app into a public sandbox where **nothing leaves the server**. It is off by default;
without it the app behaves exactly as described above.

| With `DEMO_MODE=true` | |
|---|---|
| LLM | forced to `mock` (deterministic engine); Gemini/Groq keys dropped; embeddings `local` |
| Stripe | dashboard and refund mirror off; `STRIPE_API_KEY` dropped. Refunds exist only in the local SQLite ledger |
| Email | IMAP listener and SMTP sender off, credentials dropped. A mailbox saved earlier is not reconnected, and connecting or testing one from the cockpit returns `403`. Customer replies and supervisor alerts are recorded (outbox / `simulated`), never sent |
| Tracing | LangSmith off |
| Cockpit | banner *"Demo mode: simulated payments and email"*; five one-click scenarios (duplicate charge, fraud, prompt injection, outside the refund window, Spanish) |
| Abuse limits | per client IP: `DEMO_RATE_LIMIT_PER_MINUTE` (all requests, default 300) and `DEMO_WRITE_LIMIT_PER_MINUTE` (POST/PUT/PATCH/DELETE, default 20) → `429` with `Retry-After`; request bodies above `DEMO_MAX_BODY_BYTES` (64 KB) → `413`. `/health` is exempt |

These are enforced in `Settings` itself (`app/config.py`), so a demo deployment that inherits real keys by
mistake still cannot use them. `tests/test_demo_mode.py` runs the scenarios, the email simulator and the inbound
webhook with every integration configured and fails if any non-loopback connection or DNS lookup is attempted.

**Client IP for rate limiting.** Behind a proxy, the last `X-Forwarded-For` entry is the proxy itself, and the
entries to its left can be forged by the client. Set `DEMO_CLIENT_IP_HEADERS` to the headers your edge sets.
Render sits behind Cloudflare, so `render.yaml` uses `true-client-ip,cf-connecting-ip`. Otherwise
`DEMO_TRUSTED_PROXY_HOPS` (default 1) selects the N-th `X-Forwarded-For` entry from the right.

**Render** ([`render.yaml`](render.yaml)) deploys a free Python web service:
- `uvicorn app.main:app --host 0.0.0.0 --port $PORT`;
- health check `GET /health`, which returns 200 once the runtime is up;
- `DEMO_MODE=true`.

The SQLite ledger is seeded with the demo customers at startup. The free plan's filesystem is ephemeral, so
every restart, deploy or 15-minute idle spin-down starts from a clean demo.

## Author

**Santino Genta** — design and development of SentinelOps.
