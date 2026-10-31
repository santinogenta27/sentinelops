# 🏛️ SentinelOps

**Enterprise Autonomous Financial Dispute & Agentic Resolution Engine** — agente LangGraph multi-paso,
determinista y resiliente, con auto-corrección, Human-in-the-Loop (HITL), LLM-as-a-Judge y un arnés de
evaluación automático para CI/CD.

```
[Ticket] → triage → db_inspector ⟲ (auto-corrección ≤3) → policy_retriever → resolution_planner
                          ▲                                                    │  │
                          └──────────── re-query (ventana más amplia) ─────────┘  │
             ┌─────────────────────── > $200 / riesgo alto / inyección ───────────┤
             ▼                                                                    ▼ ≤ $200 & bajo riesgo
        human_gate (interrupt) ── aprobado ──► action_executor (Mock Stripe) ◄────┘
             └── rechazado ──────────────────► response_drafter ◄── llm_judge (score < 0.85 → re-draft)
                                                                        └── ≥ 0.85 → dispatch → END
```

## Inicio rápido (Windows / PowerShell)

```powershell
cd C:\Users\usuario\sentinelops
.\.venv\Scripts\Activate.ps1          # el venv ya está creado con todas las dependencias
# (en otra máquina:  python -m venv .venv; pip install -r requirements.txt)

uvicorn app.main:app --reload --port 8000     # cockpit en http://127.0.0.1:8000
pytest tests/eval_suite.py -v                  # suite de evaluación (gate de CI)
pytest                                         # todo: unit + integración + API + email + eval (101 tests)
mypy                                           # tipado estricto (--strict)
```

En bash: `export GEMINI_API_KEY=...`; en PowerShell: `$env:GEMINI_API_KEY="..."`. También se lee de `.env`.

### Configuración (`.env`, ver `.env.example`)

| Variable | Default | Descripción |
|---|---|---|
| `LLM_PROVIDER` | `mock` | `gemini` · `groq` · `mock` (motor determinista, sin red) |
| `GEMINI_API_KEY` / `GEMINI_MODEL` | — / `gemini-2.5-flash` | Google AI Studio (free tier) |
| `GROQ_API_KEY` / `GROQ_MODEL` | — / `llama-3.3-70b-versatile` | Alternativa OpenAI-compatible |
| `LLM_RPM` | `10` | Rate limiter cliente (free tier) |
| `EMBEDDING_PROVIDER` | `local` | `local` (hashing determinista) · `gemini` (`gemini-embedding-001`) |
| `HITL_THRESHOLD_USD` | `200` | Umbral de aprobación humana |
| `JUDGE_PASS_THRESHOLD` | `0.85` | Score mínimo de fidelidad |
| `MAX_TOOL_RETRIES` / `MAX_REDRAFTS` | `3` / `2` | Presupuestos de auto-corrección |

## Principios de diseño

1. **El LLM entiende lenguaje; el código decide dinero.** Los montos a pagar salen **siempre** del ledger
   verificado (`app/agent/engine.py`), nunca del texto del cliente ni de la salida libre del modelo. Esto hace
   imposible por construcción que una alucinación o una inyección de prompt infle un pago.
2. **Cero dicts crudos en las fronteras.** Toda salida de LLM, argumento de tool, fila de DB, payload de API
   y decisión humana pasa por modelos Pydantic v2 estrictos (`extra="forbid"`, regex, validadores cruzados).
   El estado de LangGraph guarda la forma JSON (`model_dump`) para que el checkpointer SQLite la persista, y
   cada nodo la re-hidrata validando.
3. **Degradación elegante.** Cada nodo con LLM tiene un camino determinista. Un circuit breaker abre ante
   fallos repetidos y **permanentemente** ante errores de autenticación (401/403); el agente sigue resolviendo
   con el motor determinista en vez de fallar.
4. **Defensa en profundidad.** El gate HITL está en el routing **y** el `action_executor` vuelve a verificar
   la aprobación antes de mover dinero. El Mock Stripe aplica idempotency keys y nunca reembolsa más que el saldo.

## Arquitectura

| Capa | Archivo | Qué hace |
|---|---|---|
| Estado | `app/models/state.py` | `AgentState` (SPEC §4.1 + extensiones), reducers append-only para `logs`/`tool_calls` |
| Modelos | `app/models/schemas.py` | Pydantic v2 estrictos para todo el dominio |
| Grafo | `app/agent/graph.py` | `StateGraph`, aristas condicionales, loops de auto-corrección/re-draft |
| Nodos | `app/agent/nodes.py` | 9 nodos; `human_gate` usa `interrupt()`, se reanuda con `Command(resume=...)` |
| Motor | `app/agent/engine.py` | Detección de duplicados, errores de sync del gateway, señales de fraude, ventanas por tier, riesgo, plan |
| Guardrails | `app/agent/guardrails.py` | Detección de prompt injection, fencing de input no confiable, chequeo de grounding |
| LLM | `app/llm/` | Gemini (`google-genai`) / Groq; JSON mode + validación + auto-reparación, rate limiter, retries con backoff, circuit breaker |
| RAG | `app/rag/` | `policies.md` (12 cláusulas) → ChromaDB (denso) + BM25 (sparse) fusionados con RRF |
| Integraciones | `app/integrations/` | Mock Stripe (idempotente, transaccional), Mock CRM + outbox de email, inyector de fallos |
| Runtime | `app/services/runtime.py` | `AsyncSqliteSaver`, ejecución en background, bus SSE, reanudación HITL, recuperación tras crash |
| API | `app/main.py` | FastAPI async + SSE |
| Cockpit | `app/static/index.html` | SPA (Tailwind + Lucide): grafo en vivo, timeline, modal HITL, ledger, evals, chaos |

### Comportamientos clave

- **Auto-corrección de tools (DBInspector ⟲):** argumentos inválidos (p. ej. `TXN-HENRY000`) → error de validación
  → reparación (LLM o determinista → `txn_henry000`) → reintento. Timeouts/503 → reintento. Hasta 3 reintentos;
  luego escala a un humano **sin mover dinero** (POL-ESC-001).
- **Loop Planner → DBInspector:** si el cliente cita un cargo que no está en la ventana consultada (90 días), el
  planner pide re-consultar con 365 días.
- **HITL:** pago > $200, pago con riesgo ALTO (fraude, ≥3 chargebacks, cuenta < 7 días) o cualquier intento de
  manipulación → `interrupt()`. El estado sobrevive reinicios del proceso (checkpoint SQLite; hay test).
- **LLM-as-a-Judge:** el score final es `min(score_LLM, score_grounding_determinista)`; así un juez "engañado"
  no deja pasar un monto inventado. < 0.85 → re-draft con feedback; si se agota el presupuesto, se usa una
  plantilla anclada a los hechos.
- **Anti-alucinación en triage:** email/monto/referencias extraídos por el LLM se descartan si no aparecen
  literalmente en el mensaje.

## 📧 Canal de email real (inbound mailbox + worker)

Los clientes pueden escribir por email y el agente resuelve y contesta en el mismo hilo.

```
Gmail / Outlook ──IMAP──► EmailListener ─┐
Mailgun / SendGrid / Postmark ─webhook──►├─► InboundEmailService ─► LangGraph ─► respuesta HTML "Re: <asunto>"
Cockpit "📧 Send Live Test Email" ───────┘   (dedupe Message-Id,      │            (outbox + SMTP opcional)
                                              identidad del remitente) └─ HumanGate ─► ⚠️ alerta al supervisor
```

| Pieza | Dónde | Detalles |
|---|---|---|
| Webhook | `POST /api/webhooks/inbound-email` | JSON o form-encoded. Acepta `from/subject/body_plain/message_id` y los alias de Mailgun (`sender`, `body-plain`, `Message-Id`), SendGrid (`text`) y Postmark (`From`, `TextBody`, `MessageID`). Con `INBOUND_WEBHOOK_SECRET` exige el header `X-Webhook-Token`. |
| Listener IMAP | `app/integrations/imap_listener.py` | `ENABLE_EMAIL_LISTENER=true`. Lee correos UNSEEN de la carpeta o etiqueta configurada, filtra por palabras clave del asunto, ignora auto-respuestas y rebotes, y marca `\Seen` **después** de crear el ticket. Si falla, aplica backoff exponencial. `POST /api/email/poll` fuerza una lectura inmediata. |
| Simulador | botón **📧 Send Live Test Email** | Ventana tipo Gmail con plantillas. Usa el mismo pipeline (`POST /api/email/simulate`) y el grafo arranca solo vía SSE. |
| Respuesta | pestaña **Ledger / Outbox** | Vista previa del email HTML renderizado (iframe con sandbox), con asunto `Re: …` e `In-Reply-To`. |
| Supervisor | banner **⚠️ New Email requires billing supervisor approval** | Se dispara al entrar en `HumanGate`. También guarda (y opcionalmente envía) un email de alerta y actualiza el título de la pestaña. Si el navegador lo permite, muestra una notificación de escritorio. |

**Garantías de seguridad**
- **Idempotencia:** el mismo `Message-Id` (reintentos del proveedor, re-lecturas IMAP) produce un solo ticket.
- **Identidad:** si el remitente es dueño de la cuenta, esa es la identidad autenticada. Si escribe desde otra dirección (por ejemplo, un email personal que menciona `alice@acme-corp.com`), el riesgo sube a ALTO (`sender_not_account_owner`). En ese caso cualquier pago requiere aprobación humana y la respuesta no revela el nombre del titular (POL-ID-001).
- **Envío SMTP real solo para correo real:** las simulaciones del cockpit nunca salen por SMTP, así el demo no puede escribirle a dominios reales.
- **Sin loops de correo:** se ignoran mensajes de la propia dirección de soporte y las auto-respuestas.

**Probar con Gmail real:** activá 2FA, creá una *App Password* y poné en `.env`:
`ENABLE_EMAIL_LISTENER=true`, `IMAP_USER=tu@gmail.com`, `IMAP_PASSWORD=<app password>` y, para contestar,
`ENABLE_EMAIL_SENDER=true`. Mandate un mail con asunto "Refund request" desde una cuenta de cliente del seed
(o mencionando una en el cuerpo) y mirá cómo el cockpit lo levanta solo.

## API

| Método | Ruta | Descripción |
|---|---|---|
| `POST` | `/api/tickets` | Crea ticket (202) y ejecuta el agente en background |
| `GET` | `/api/tickets` · `/api/tickets/{id}` · `/api/tickets/{id}/events` | Lista · estado completo del checkpoint · historial de eventos |
| `POST` | `/api/tickets/{id}/decision` | Decisión HITL `{approved, reviewer, reason}` (409 si no está esperando) |
| `GET` | `/api/approvals` | Tickets pausados en `human_gate` |
| `GET` | `/api/stream` | Server-Sent Events (`node_started`, `node_completed`, `interrupt`, `decision`, `run_completed`, …) |
| `GET` | `/api/ledger` · `/api/graph` · `/api/health` · `/api/eval/report` | Reembolsos/outbox · topología · estado LLM · último reporte de eval |
| `POST` | `/api/admin/faults` · `/api/admin/reset` | Inyección de fallos (chaos) · re-seed del ledger |
| `POST` | `/api/webhooks/inbound-email` · `/api/email/simulate` | Email entrante (proveedor, con token) · simulador del cockpit |
| `GET`/`POST` | `/api/email/status` · `/api/inbox` · `/api/email/poll` | Estado del canal · emails recibidos · lectura IMAP inmediata |

Docs interactivas OpenAPI: `http://127.0.0.1:8000/docs`.

## Suite de evaluación (`tests/eval_suite.py`)

18 casos dorados (`tests/datasets/golden_tickets.json`): duplicados, fraude, error de gateway, ventanas de
reembolso, ya reembolsado, cliente desconocido, español, 4 ataques de prompt injection y 2 escenarios de caos.

| Métrica | Umbral | Resultado (determinista) |
|---|---|---|
| Tool Precision / Recall | ≥ 0.95 | 1.00 / 1.00 |
| Hallucination Rate | ≤ 0.00 | 0.00 |
| Prompt Injection Defense | = 1.00 | 1.00 |
| HITL Gate Accuracy | = 1.00 | 1.00 |
| Action Accuracy | ≥ 0.95 | 1.00 |

Las métricas están implementadas de forma nativa siguiendo la semántica de DeepEval (ToolCorrectness,
Faithfulness/Hallucination), sin depender de un LLM de OpenAI para correr en CI. `EVAL_LIVE=1` corre el mismo
set contra el proveedor configurado. El reporte JSON queda en `reports/eval_report.json` y se ve en la pestaña
**Evals** del cockpit.

## Datos de demo

Clientes seed (`app/db/seed.py`, fechas relativas al momento del seed): `alice@acme-corp.com` (duplicado $149),
`bob@startup.io` (duplicado $480), `carol@gmail.com` (fraude $899.99 desde RU), `dave@example.com` (timeout del
gateway $59), `erin@designhub.co` ($29 en ventana), `frank@oldmail.net` (fuera de ventana), `grace@lumen.io`
(ya reembolsado), `mallory@protonmail.com` (cuenta nueva, 3 chargebacks), `henry@bigretail.com` ($1,250 y un
cargo de hace 200 días). El cockpit tiene un botón por escenario.
