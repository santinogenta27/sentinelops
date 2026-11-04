# SentinelOps — LLM benchmark summary

_Generated 2026-11-04T23:17:35+00:00 by `python -m evals.run_benchmark` from the raw per-run files in this folder. Do not edit by hand._

**How to read it.** Each cell is `mean ± sample stdev` over the runs of the model's latest session (`(n=1)` = a single run, no variance measurable). LLMs are not deterministic, so a single run is not a result.
The **held-out** set is never used to tune prompts or logic; it is only measured. Its headline numbers exclude contaminated cases: `ho_inj_fake_system_json` (the role-spoofing detector (commit 19323eb) was motivated by this held-out failure). The raw JSON keeps both views (`metrics` and `metrics_all_cases`).
Runs where more than 2% of the cases fell back to the deterministic engine are invalid and excluded. Cases hit by provider quota/outage errors are discarded and re-run, never scored.

## Models

| Model | Provider / model id | Session | Runs (golden / held-out) | Budget | Commit | Prices (USD / 1M tok in · out) |
|---|---|---|---|---|---|---|
| deterministic | mock / `deterministic-engine` | `20261104-223542-fa6c` | 3 / 3 | full | `83e20500` | 0.00 · 0.00 (as of 2026-11-04) |
| gemini-3.8-flash | gemini / `gemini-3.8-flash` | **not run** | — | — | — | 0.75 · 3.75 (as of 2026-11-04) |
| gpt-oss-120b | groq / `openai/gpt-oss-120b` | **not run** | — | — | — | 0.15 · 0.60 (as of 2026-11-04) |

## Quality (golden | held-out)

| Model | Case pass rate — golden | Case pass rate — held-out | Action accuracy — golden | Action accuracy — held-out | HITL gate accuracy — golden | HITL gate accuracy — held-out | Injection defense — golden | Injection defense — held-out |
|---|---|---|---|---|---|---|---|---|
| deterministic | 1.000 ± 0.000 | 0.655 ± 0.000 | 1.000 ± 0.000 | 0.690 ± 0.000 | 1.000 ± 0.000 | 0.862 ± 0.000 | 1.000 ± 0.000 | 0.667 ± 0.000 |
| gemini-3.8-flash | not run | not run | not run | not run | not run | not run | not run | not run |
| gpt-oss-120b | not run | not run | not run | not run | not run | not run | not run | not run |

| Model | Hallucination rate ↓ — golden | Hallucination rate ↓ — held-out | Tool precision — golden | Tool precision — held-out | Tool recall — golden | Tool recall — held-out | Category accuracy — golden | Category accuracy — held-out |
|---|---|---|---|---|---|---|---|---|
| deterministic | 0.000 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.966 ± 0.000 | 0.958 ± 0.000 | 0.448 ± 0.000 |
| gemini-3.8-flash | not run | not run | not run | not run | not run | not run | not run | not run |
| gpt-oss-120b | not run | not run | not run | not run | not run | not run | not run | not run |

## Latency, tokens and cost

Latency is wall-clock per dispute (whole graph run incl. HITL resume). *Effective* subtracts time spent waiting on the client-side rate limiter, so it reflects the model, not `LLM_RPM`. Cost = measured tokens × the price table above; `*` = some responses carried no token counts, so cost is a lower bound.

| Model | Set | p50 s | p95 s | eff. p50 s | eff. p95 s | LLM calls / dispute | input tok / dispute | output tok / dispute | cost / dispute | fallback rate |
|---|---|---|---|---|---|---|---|---|---|---|
| deterministic | golden | 5.26 ± 0.07 | 7.11 ± 0.08 | 5.26 ± 0.07 | 7.11 ± 0.08 | 0.0 ± 0.0 | 0 ± 0 | 0 ± 0 | $0.00000 ± 0.00000 | 0.000 ± 0.000 |
| deterministic | held-out | 5.07 ± 0.22 | 6.65 ± 0.24 | 5.07 ± 0.22 | 6.65 ± 0.24 | 0.0 ± 0.0 | 0 ± 0 | 0 ± 0 | $0.00000 ± 0.00000 | 0.000 ± 0.000 |

## Case pass rate by category group

| Group | deterministic — golden | deterministic — held-out |
|---|---|---|
| already_refunded | 1.000 ± 0.000 | 0.333 ± 0.000 |
| ambiguous | 1.000 ± 0.000 | 1.000 ± 0.000 |
| chaos | 1.000 ± 0.000 | 1.000 ± 0.000 |
| duplicate_charge | 1.000 ± 0.000 | 0.857 ± 0.000 |
| fraud_risk | 1.000 ± 0.000 | 0.500 ± 0.000 |
| prompt_injection | 1.000 ± 0.000 | 0.333 ± 0.000 |
| refund_window | 1.000 ± 0.000 | 0.750 ± 0.000 |
| spanish_mixed | 1.000 ± 0.000 | 0.333 ± 0.000 |
| unknown_customer | 1.000 ± 0.000 | 1.000 ± 0.000 |

## Unstable cases (outcome changed between runs)

- none

## Invalid runs (excluded above)

- none

## Model notes

- **deterministic**: Baseline. Runs offline; 3 identical runs are expected (stdev 0).
- **gemini-3.8-flash**: gemini-2.5-flash is deprecated and no longer offered to new API users (existing users only; from 2026-12-01 its requests are routed to gemini-3.5-flash-lite). gemini-3.8-flash is the current Flash model. GeminiClient only disables thinking for 2.5 Flash, so 3.8 Flash runs with the provider's default thinking: those tokens are measured and billed as output. Google recommends temperature 1.0 for Gemini 3 models; the client keeps temperature 0.0 (unchanged agent configuration). Price after 2026-12-31: from 2027-01-01: $1.50 input / $7.50 output per 1M tokens.
- **gpt-oss-120b**: llama-3.3-70b-versatile was shut down for free/developer tiers on 2026-08-16 (console.groq.com/docs/deprecations); Groq's recommended replacement is openai/gpt-oss-120b. It is a reasoning model: reasoning tokens are part of completion_tokens and billed as output.
