"""SentinelOps model benchmark: golden + held-out sets against real LLMs, with the deterministic engine as baseline.

    python -m evals.run_benchmark --estimate                      # cost/time of both presets, no API call
    python -m evals.run_benchmark --smoke                         # 5 cases per set, 1 run: measures tokens/case
    python -m evals.run_benchmark                                 # full preset: golden x3 + held-out x3 per model
    python -m evals.run_benchmark --budget free                   # free tiers: golden x1 + held-out x3, paced
    python -m evals.run_benchmark --resume latest [--wait]        # continue a paused/interrupted session
    python -m evals.run_benchmark --summarize-only                # rebuild SUMMARY.md from the raw JSON files
    python -m evals.run_benchmark --models deterministic          # baseline only (offline)

Models, prices and free-tier quotas live in evals/models.toml. Raw results go to
reports/benchmarks/<YYYYMMDD-HHMMSS>_<model>_r<k>.json (one file per run, both sets) and the comparison
table to reports/benchmarks/SUMMARY.md.

Honesty rules enforced here:
  * a model whose API key is missing is never run and never reported: the script exits (code 2) first;
  * every case is run with the real model or not recorded at all: provider failures (quota, 429, outage)
    make the agent silently fall back to its deterministic engine, so such cases are discarded and retried
    instead of being scored as the model's; fallbacks caused by the model's own answers are counted and a
    run with too many of them is marked invalid;
  * the held-out set is only measured; nothing here feeds back into prompts or agent logic.
"""

from __future__ import annotations

import os

# Hermetic environment, same as tests/conftest.py and tests/heldout_eval.py: never touch the real Stripe
# account, keep Auto-Pilot OFF (the harness takes every HITL decision), no real email.
os.environ["STRIPE_API_KEY"] = ""
os.environ["AUTOPILOT_DEFAULT"] = "false"
os.environ["ENABLE_EMAIL_LISTENER"] = "false"
os.environ["ENABLE_EMAIL_SENDER"] = "false"

import argparse  # noqa: E402
import asyncio  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from collections import deque  # noqa: E402
from collections.abc import Awaitable, Callable, Sequence  # noqa: E402
from dataclasses import asdict, dataclass, field  # noqa: E402
from datetime import datetime, timedelta  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import httpx  # noqa: E402

from app.config import PROJECT_ROOT, Settings  # noqa: E402
from app.integrations.faults import FaultInjector  # noqa: E402
from app.llm.base import LLMClient, LLMError, LLMTask, LLMUsage, RateLimiter  # noqa: E402
from app.llm.factory import build_llm  # noqa: E402
from app.services.runtime import SentinelRuntime  # noqa: E402
from evals.config import BenchmarkConfig, ModelSpec, load_config  # noqa: E402
from evals.session import Session, all_measurements, iso, next_reset, quota_day_start, usage_since, utc_now  # noqa: E402
from evals.stats import summarize_set  # noqa: E402
from evals.summary import write_summary  # noqa: E402
from tests.eval_suite import CASES as GOLDEN_CASES  # noqa: E402
from tests.eval_suite import DATASET as GOLDEN_PATH  # noqa: E402
from tests.eval_suite import _run_case  # noqa: E402
from tests.heldout_eval import DATASET as HELDOUT_PATH  # noqa: E402
from tests.heldout_eval import preflight as heldout_preflight  # noqa: E402

ROOT = PROJECT_ROOT / "reports" / "benchmarks"
RUN_SCHEMA = "sentinelops.benchmark.run/1"
DATASETS = ("golden", "heldout")
PRESETS: dict[str, dict[str, int]] = {"full": {"golden": 3, "heldout": 3}, "free": {"golden": 1, "heldout": 3}}
DAILY_MARGIN = 0.9  # use at most 90% of a daily free-tier quota
MAX_CONSECUTIVE_DISCARDS = 3

EXIT_CONFIG, EXIT_PREFLIGHT, EXIT_PAUSED, EXIT_INVALID = 2, 3, 4, 5
_DAILY_QUOTA_RE = re.compile(r"per ?day|perday|\bRPD\b|\bTPD\b|daily", re.IGNORECASE)


class BenchmarkError(Exception):
    def __init__(self, message: str, code: int) -> None:
        super().__init__(message)
        self.code = code


class QuotaPaused(Exception):
    def __init__(self, model_id: str, message: str, resume_after: datetime | None) -> None:
        super().__init__(message)
        self.model_id = model_id
        self.resume_after = resume_after


class ProviderUnstable(Exception):
    pass


def is_daily_quota_error(message: str | None) -> bool:
    return bool(message and _DAILY_QUOTA_RE.search(message))


# ============================================================================ pacing for free tiers
class BudgetLimiter(RateLimiter):
    """RateLimiter that also keeps a rolling tokens-per-minute window under the free-tier TPM quota."""

    def __init__(
        self, rpm: int, tpm: int, *, default_tokens_per_call: int = 2500, sleep: Callable[[float], Awaitable[None]] = asyncio.sleep
    ) -> None:
        super().__init__(rpm)
        self.tpm = tpm
        self._sleep = sleep
        self._window: deque[tuple[float, int]] = deque()
        self._seen_calls = 0
        self._seen_tokens = 0
        self._default = default_tokens_per_call

    def _estimate(self) -> int:
        return round(self._seen_tokens / self._seen_calls) if self._seen_calls else self._default

    async def acquire(self) -> None:
        await super().acquire()
        if self.tpm <= 0:
            return
        while True:
            now = time.monotonic()
            while self._window and now - self._window[0][0] > 60.0:
                self._window.popleft()
            used = sum(t for _, t in self._window)
            if not self._window or used + self._estimate() <= self.tpm:
                return
            await self._sleep(60.0 - (now - self._window[0][0]) + 0.05)

    def on_tokens(self, tokens: int) -> None:
        self._window.append((time.monotonic(), tokens))
        self._seen_calls += 1
        self._seen_tokens += tokens


# ============================================================================ helpers
def git_info() -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return ""

    commit = run("rev-parse", "HEAD") or "unknown"
    return {"commit": commit, "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def load_datasets() -> dict[str, list[dict[str, Any]]]:
    heldout: list[dict[str, Any]] = json.loads(HELDOUT_PATH.read_text(encoding="utf-8"))
    heldout_preflight(heldout)
    return {"golden": list(GOLDEN_CASES), "heldout": heldout}


def stratified_sample(cases: Sequence[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    """First ``n`` cases taken round-robin across category groups (keeps dataset order inside a group)."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for c in cases:
        groups.setdefault(c["category_group"], []).append(c)
    out: list[dict[str, Any]] = []
    while len(out) < min(n, len(cases)):
        for g in list(groups):
            if groups[g] and len(out) < n:
                out.append(groups[g].pop(0))
    return out


def env_settings() -> Settings:
    """Developer settings (.env + process env): only used for API keys and client knobs (RPM, timeout, retries)."""
    return Settings()


def bench_settings(spec: ModelSpec, data_dir: Path, env: Settings, *, trace: bool) -> Settings:
    """Hermetic settings: business rules at their defaults for every model, so models are comparable."""
    return Settings(  # type: ignore[call-arg]
        _env_file=None,
        llm_provider=spec.provider,
        gemini_api_key=env.gemini_api_key,
        gemini_model=spec.model if spec.provider == "gemini" else env.gemini_model,
        groq_api_key=env.groq_api_key,
        groq_model=spec.model if spec.provider == "groq" else env.groq_model,
        llm_rpm=env.llm_rpm,
        llm_timeout_s=env.llm_timeout_s,
        llm_max_retries=env.llm_max_retries,
        langsmith_tracing=trace and env.langsmith_tracing,
        langsmith_api_key=env.langsmith_api_key,
        langsmith_project=env.langsmith_project,
        langsmith_endpoint=env.langsmith_endpoint,
        data_dir=data_dir,
    )


def api_key(spec: ModelSpec, env: Settings) -> str:
    secret = {"gemini": env.gemini_api_key, "groq": env.groq_api_key}.get(spec.provider)
    return secret.get_secret_value().strip() if secret else ""


# ============================================================================ preflight
def check_keys(specs: Sequence[ModelSpec], env: Settings) -> None:
    missing = [s for s in specs if s.is_llm and not api_key(s, env)]
    if not missing:
        return
    have = [s.id for s in specs if s not in missing]
    lines = [f"{s.api_key_env} is not set: cannot benchmark {s.id} ({s.provider}:{s.model})." for s in missing]
    lines.append("No model was run and no result was written. Set the key(s) in .env or the environment")
    lines.append(f"or restrict the run, e.g. --models {','.join(have) or 'deterministic'}")
    raise BenchmarkError("\n".join(lines), EXIT_CONFIG)


async def check_model_available(spec: ModelSpec, key: str) -> None:
    """Fail fast (no generation, no quota used) when the model id does not exist for this key."""
    hint = f" (it replaces {spec.replaces})" if spec.replaces else ""
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            if spec.provider == "groq":
                resp = await client.get("https://api.groq.com/openai/v1/models", headers={"Authorization": f"Bearer {key}"})
                if resp.status_code in (401, 403):
                    raise BenchmarkError(f"{spec.id}: Groq rejected {spec.api_key_env} (HTTP {resp.status_code})", EXIT_PREFLIGHT)
                resp.raise_for_status()
                ids = {m.get("id") for m in resp.json().get("data", [])}
                if spec.model not in ids:
                    raise BenchmarkError(
                        f"{spec.id}: model {spec.model!r}{hint} is not offered to this Groq key. "
                        f"Pick a current id from https://console.groq.com/docs/models and update evals/models.toml.",
                        EXIT_PREFLIGHT,
                    )
            elif spec.provider == "gemini":
                resp = await client.get(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{spec.model}", headers={"x-goog-api-key": key}
                )
                if resp.status_code in (401, 403) or (resp.status_code == 400 and "API key" in resp.text):
                    raise BenchmarkError(f"{spec.id}: Google rejected {spec.api_key_env} (HTTP {resp.status_code})", EXIT_PREFLIGHT)
                if resp.status_code == 404:
                    raise BenchmarkError(
                        f"{spec.id}: model {spec.model!r}{hint} is not available to this Gemini key. "
                        f"Pick a current id from https://aistudio.google.com/docs/models and update evals/models.toml.",
                        EXIT_PREFLIGHT,
                    )
                resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise BenchmarkError(f"{spec.id}: could not reach the {spec.provider} API to check the model: {exc}", EXIT_PREFLIGHT) from exc


async def probe(spec: ModelSpec, llm: LLMClient, session: Session) -> None:
    """One tiny real generation: proves the key has quota before any case is run."""
    before = llm.usage.snapshot()
    try:
        await llm.generate_text(task=LLMTask.TRIAGE, system="Health check.", prompt="Reply with the single word OK.")
    except LLMError as exc:
        delta = llm.usage.since(before)
        session.append({"type": "probe", "model": spec.id, "ok": False, "error": str(exc)[:300], "usage": _usage(delta)})
        if is_daily_quota_error(str(exc)) and spec.free_tier:
            raise QuotaPaused(spec.id, f"{spec.id}: daily quota already exhausted ({exc})", _reset_of(spec)) from exc
        raise BenchmarkError(f"{spec.id}: probe request failed, nothing was benchmarked: {exc}", EXIT_PREFLIGHT) from exc
    session.append({"type": "probe", "model": spec.id, "ok": True, "usage": _usage(llm.usage.since(before))})


def _usage(u: LLMUsage) -> dict[str, int]:
    return {"calls": u.calls, "input_tokens": u.input_tokens, "output_tokens": u.output_tokens}


def _reset_of(spec: ModelSpec) -> datetime | None:
    return next_reset(utc_now(), spec.free_tier.daily_reset_utc_hour) if spec.free_tier else None


# ============================================================================ running
@dataclass(slots=True)
class Ctx:
    cfg: BenchmarkConfig
    session: Session
    env: Settings
    datasets: dict[str, list[dict[str, Any]]]
    trace: bool = False
    wait: bool = False
    parallel: bool = False
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep
    llm_factory: Callable[[ModelSpec, Settings], LLMClient | None] | None = None
    paused: list[QuotaPaused] = field(default_factory=list)
    invalid: list[str] = field(default_factory=list)
    written: list[Path] = field(default_factory=list)

    @property
    def budget(self) -> str:
        return str(self.session.plan["budget"])

    def runs_for(self, spec: ModelSpec) -> dict[str, int]:
        runs: dict[str, int] = self.session.plan["runs"]
        if not spec.is_llm and not self.session.plan.get("smoke"):
            return {ds: PRESETS["full"][ds] for ds in runs}  # the free baseline always gets 3 runs per set
        return runs

    def cases_for(self, dataset: str) -> list[dict[str, Any]]:
        limit = self.session.plan.get("limit")
        cases = self.datasets[dataset]
        return stratified_sample(cases, int(limit)) if limit else cases


def make_llm(ctx: Ctx, spec: ModelSpec, settings: Settings) -> LLMClient | None:
    if not spec.is_llm:
        return None
    llm = ctx.llm_factory(spec, settings) if ctx.llm_factory else build_llm(settings)
    if llm is None:
        raise BenchmarkError(f"{spec.id}: could not build the LLM client (missing key?)", EXIT_CONFIG)
    if ctx.budget == "free" and spec.free_tier:
        ft = spec.free_tier
        llm.limiter = BudgetLimiter(min(settings.llm_rpm, ft.rpm), int(ft.tpm * DAILY_MARGIN) if ft.tpm else 0)
    return llm


def check_daily_budget(ctx: Ctx, spec: ModelSpec) -> None:
    """Free budget: stop *before* a case that would not fit in what is left of today's quota."""
    ft = spec.free_tier
    if ctx.budget != "free" or ft is None or not spec.is_llm:
        return
    now = utc_now()
    used = usage_since(ctx.session.root, spec.id, quota_day_start(now, ft.daily_reset_utc_hour))
    m = all_measurements(ctx.session.root, spec.id)
    est_calls = m["calls"] if m else ctx.cfg.est_calls_per_case
    est_tokens = (
        m["input_tokens"] + m["output_tokens"]
        if m
        else est_calls * (ctx.cfg.est_input_tokens_per_call + spec.est_output_tokens_per_call)
    )
    reason = None
    if ft.rpd and used["calls"] + est_calls > ft.rpd * DAILY_MARGIN:
        reason = f"{used['calls']}/{ft.rpd} requests used today"
    elif ft.tpd and used["tokens"] + est_tokens > ft.tpd * DAILY_MARGIN:
        reason = f"{used['tokens']:,}/{ft.tpd:,} tokens used today"
    if reason:
        raise QuotaPaused(spec.id, f"{spec.id}: daily free-tier budget reached ({reason})", next_reset(now, ft.daily_reset_utc_hour))


async def run_case(rt: SentinelRuntime, faults: FaultInjector, case: dict[str, Any], dataset: str, llm: LLMClient | None) -> tuple[dict[str, Any], LLMUsage]:
    before = llm.usage.snapshot() if llm else LLMUsage()
    t0 = time.perf_counter()
    result = await _run_case(rt, faults, case, heldout=dataset == "heldout")
    wall = time.perf_counter() - t0
    delta = llm.usage.since(before) if llm else LLMUsage()
    record = asdict(result)
    record["perf"] = {
        "wall_s": round(wall, 4),
        "effective_s": round(max(wall - delta.limiter_wait_seconds, 0.0), 4),
        "limiter_wait_s": round(delta.limiter_wait_seconds, 4),
        "provider_s": round(delta.provider_seconds, 4),
        "calls": delta.calls,
        "retries": delta.retries,
        "rate_limited_attempts": delta.rate_limited_attempts,
        "input_tokens": delta.input_tokens,
        "output_tokens": delta.output_tokens,
        "calls_without_usage": delta.calls_without_usage,
        "model_fallbacks": delta.surfaced_model_errors,
    }
    record["llm_degraded"] = delta.surfaced_model_errors > 0
    return record, delta


async def run_set(ctx: Ctx, spec: ModelSpec, dataset: str, run: int) -> None:
    session = ctx.session
    done = session.completed(spec.id, dataset, run)
    pending = [c for c in ctx.cases_for(dataset) if c["id"] not in done]
    if not pending:
        return
    print(f"[{spec.id}] {dataset} run {run}: {len(pending)} case(s) to run ({len(done)} already done)", flush=True)
    with tempfile.TemporaryDirectory(prefix="sentinel-bench-", ignore_cleanup_errors=True) as tmp:
        settings = bench_settings(spec, Path(tmp), ctx.env, trace=ctx.trace)
        llm = make_llm(ctx, spec, settings)
        faults = FaultInjector()
        # With --parallel each runtime needs its own Chroma store (persist_vectors=True, in its temp dir): the
        # ephemeral client is a process-wide singleton whose "policies" collection would be dropped by a runtime
        # created concurrently. Sequential runs never overlap, so they keep the faster in-memory store.
        rt = await SentinelRuntime.create(settings, llm=llm, faults=faults, persist_vectors=ctx.parallel)
        try:
            if llm is not None and not session.plan.get("probed", {}).get(spec.id):
                await probe(spec, llm, session)
                session.plan.setdefault("probed", {})[spec.id] = True
            discards = 0
            i = 0
            while i < len(pending):
                case = pending[i]
                check_daily_budget(ctx, spec)
                if llm is not None and not llm.breaker.allow() and llm.breaker.state != "open:permanent":
                    # While the circuit is open the nodes skip the LLM *silently* (SentinelNodes._llm returns
                    # None): never start a case before the cooldown has elapsed.
                    await ctx.sleep(llm.breaker.cooldown_s + 1)
                record, delta = await run_case(rt, faults, case, dataset, llm)
                base = {"model": spec.id, "dataset": dataset, "run": run, "case_id": case["id"]}
                if llm is not None and llm.breaker.state == "open:permanent":
                    session.append({"type": "discard", **base, "reason": llm.breaker.reason, "usage": _usage(delta)})
                    raise ProviderUnstable(f"{spec.id}: provider permanently unavailable: {llm.breaker.reason}")
                skipped = llm is not None and delta.calls == 0  # no request reached the model at all
                if delta.surfaced_infra_errors or skipped:
                    # The agent fell back to its deterministic engine because the *provider* failed (or the
                    # circuit was open): this is not the model's result. Log the spend, never score it, retry later.
                    reason = delta.last_infra_error or "no request reached the model (circuit breaker open)"
                    delta.last_infra_error = reason
                    session.append({"type": "discard", **base, "reason": reason, "usage": _usage(delta)})
                    if is_daily_quota_error(delta.last_infra_error):
                        raise QuotaPaused(spec.id, f"{spec.id}: provider daily quota hit ({delta.last_infra_error})", _reset_of(spec))
                    discards += 1
                    if discards >= MAX_CONSECUTIVE_DISCARDS:
                        raise ProviderUnstable(
                            f"{spec.id}: {discards} consecutive cases hit provider errors (last: {delta.last_infra_error}). "
                            "The session is saved; retry later with --resume latest."
                        )
                    backoff = min(60.0 * 2 ** (discards - 1), 300.0)
                    print(f"[{spec.id}] provider error on {case['id']}, case discarded; retrying in {backoff:.0f}s", flush=True)
                    await ctx.sleep(backoff)
                    continue
                discards = 0
                session.append({"type": "case", **base, "record": record, **({"usage": _usage(delta)} if llm else {})})
                mark = "PASS" if record["passed"] else "FAIL"
                print(f"[{spec.id}] {dataset} r{run} {i + 1}/{len(pending)} {case['id']}: {mark}", flush=True)
                i += 1
        finally:
            await rt.close()


def write_run(ctx: Ctx, spec: ModelSpec, run: int) -> Path | None:
    session = ctx.session
    if session.plan.get("smoke") or session.run_written(spec.id, run):
        return None
    sets: dict[str, Any] = {}
    reasons: list[str] = []
    for ds, n_runs in ctx.runs_for(spec).items():
        if run > n_runs:
            continue
        done = session.completed(spec.id, ds, run)
        order = [c["id"] for c in ctx.cases_for(ds)]
        if set(order) - done.keys():
            return None  # incomplete: written once the last case finishes
        records = [done[cid] for cid in order]
        contaminated = ctx.cfg.contaminated_heldout if ds == "heldout" else {}
        summary = summarize_set(records, spec, contaminated)
        summary["results"] = records
        sets[ds] = summary
        rate = summary["perf"]["fallback_rate"] or 0.0
        if rate > session.plan["max_fallback_rate"]:
            reasons.append(f"{ds}: fallback rate {rate:.1%} > {session.plan['max_fallback_rate']:.0%}")
    if not sets:
        return None
    env = ctx.env
    doc = {
        "schema": RUN_SCHEMA,
        "session_id": session.session_id,
        "run": run,
        "budget": ctx.budget,
        "generated_at": iso(utc_now()),
        "model": spec.public(),
        "git": session.plan["git"],
        "datasets": session.plan["dataset_hashes"],
        "settings": {
            "llm_rpm": env.llm_rpm,
            "llm_timeout_s": env.llm_timeout_s,
            "llm_max_retries": env.llm_max_retries,
            "business_rules": "defaults (evals ignore .env business rules)",
        },
        "valid": not reasons,
        "invalid_reason": "; ".join(reasons) or None,
        "sets": sets,
    }
    path = session.root / f"{session.session_id[:15]}_{spec.slug}_r{run}.json"
    path.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    session.append({"type": "run_written", "model": spec.id, "run": run, "path": path.name, "valid": not reasons})
    if reasons:
        ctx.invalid.append(f"{path.name}: {'; '.join(reasons)}")
    ctx.written.append(path)
    return path


async def run_model(ctx: Ctx, spec: ModelSpec) -> None:
    runs = ctx.runs_for(spec)
    datasets = [ds for ds in DATASETS if ds in runs]
    while True:
        try:
            for run in range(1, max(runs.values()) + 1):
                for ds in datasets:
                    if run <= runs[ds]:
                        await run_set(ctx, spec, ds, run)
                write_run(ctx, spec, run)
            return
        except QuotaPaused as pause:
            ctx.session.append(
                {"type": "pause", "model": spec.id, "reason": str(pause), "resume_after": iso(pause.resume_after) if pause.resume_after else None}
            )
            if not ctx.wait or pause.resume_after is None:
                ctx.paused.append(pause)
                print(f"[{spec.id}] paused: {pause}", flush=True)
                return
            delay = max((pause.resume_after - utc_now()).total_seconds(), 0) + 120
            print(f"[{spec.id}] {pause} - waiting {timedelta(seconds=round(delay))} for the quota reset (--wait)", flush=True)
            await ctx.sleep(delay)


async def run_all(ctx: Ctx, specs: Sequence[ModelSpec], *, parallel: bool) -> None:
    async def worker(group: Sequence[ModelSpec]) -> None:
        for spec in group:
            await run_model(ctx, spec)

    if not parallel:
        await worker(specs)
        return
    groups: dict[str, list[ModelSpec]] = {}
    for s in specs:
        groups.setdefault(s.provider, []).append(s)  # same provider = shared quota: sequential
    await asyncio.gather(*(worker(g) for g in groups.values()))


# ============================================================================ estimate
def estimate(
    cfg: BenchmarkConfig, specs: Sequence[ModelSpec], env: Settings, datasets: dict[str, list[dict[str, Any]]], root: Path = ROOT
) -> str:
    out = [
        "Estimated cost and duration (no API was called).",
        f"LLM_RPM={env.llm_rpm}. 'measured' = averages from previous benchmark sessions; 'assumed' = evals/models.toml defaults.",
        "",
    ]
    header = f"{'model':<18}{'preset':<7}{'cases':>6}{'LLM calls':>11}{'tokens in':>12}{'tokens out':>12}{'cost USD':>10}  duration"
    out.append(header)
    out.append("-" * (len(header) + 20))
    notes: list[str] = []
    for spec in specs:
        m = all_measurements(root, spec.id)
        if spec.is_llm and m:
            calls, tin, tout, secs = m["calls"], m["input_tokens"], m["output_tokens"], m["effective_s"]
            source = f"measured over {int(m['cases'])} cases"
        else:
            calls = cfg.est_calls_per_case if spec.is_llm else 0.0
            tin = calls * cfg.est_input_tokens_per_call
            tout = calls * spec.est_output_tokens_per_call if spec.is_llm else 0.0
            secs = calls * spec.est_seconds_per_call if spec.is_llm else (m["effective_s"] if m else 0.5)
            source = "assumed" if spec.is_llm else ("measured" if m else "assumed")
        for preset, runs in PRESETS.items():
            if not spec.is_llm and preset != "full":
                continue  # the offline baseline is identical in both presets
            plan = runs if spec.is_llm else PRESETS["full"]
            n_cases = sum(plan[ds] * len(datasets[ds]) for ds in DATASETS)
            total_calls = n_cases * calls
            cost = spec.cost_usd(round(n_cases * tin), round(n_cases * tout))
            cost_cell = f"{cost:>10.2f}"
            if preset == "free" and spec.is_llm:
                cost_cell = f"{0.0:>10.2f}"
                notes.append(f"  {spec.id}: free preset costs $0 on the free tier (same tokens on a paid key: ${cost:.2f})")
            if not spec.is_llm:
                duration = _human(n_cases * secs)
            elif preset == "full":
                per_case = max(secs, calls * 60.0 / env.llm_rpm)
                duration = _human(n_cases * per_case) + " (paid tier)"
            elif spec.free_tier is None:
                duration = "n/a (no free tier in models.toml)"
            else:
                ft = spec.free_tier
                caps = []
                if ft.rpd and calls:
                    caps.append(ft.rpd * DAILY_MARGIN / calls)
                if ft.tpd and (tin + tout):
                    caps.append(ft.tpd * DAILY_MARGIN / (tin + tout))
                per_day = min(caps) if caps else float("inf")
                per_case_min = max(
                    calls / min(env.llm_rpm, ft.rpm),
                    (tin + tout) / (ft.tpm * DAILY_MARGIN) if ft.tpm else 0.0,
                    secs / 60.0,
                )
                if per_day < 1:
                    duration = "never (one dispute exceeds the daily quota)"
                elif math.isinf(per_day):
                    duration = _human(n_cases * per_case_min * 60)
                else:
                    days = math.ceil(n_cases / per_day)
                    active = min(per_day, n_cases) * per_case_min
                    duration = f"~{days} day(s) ({per_day:.1f} disputes/day, ~{active / 60:.1f} h active per day)"
            out.append(
                f"{spec.id:<18}{preset:<7}{n_cases:>6}{total_calls:>11.0f}{n_cases * tin:>12,.0f}{n_cases * tout:>12,.0f}"
                f"{cost_cell}  {duration}"
            )
        notes.append(f"  {spec.id}: per dispute {calls:.1f} calls, {tin:,.0f} in / {tout:,.0f} out tokens ({source})")
        if spec.price_after:
            notes.append(f"  {spec.id}: prices valid until {spec.price_valid_until}; {spec.price_after}")
        if spec.free_tier:
            notes.append(f"  {spec.id}: free tier {spec.free_tier.rpm} RPM / {spec.free_tier.rpd} RPD / {spec.free_tier.tpm} TPM / "
                         f"{spec.free_tier.tpd or '-'} TPD - source: {spec.free_tier.quota_source}")
    out += ["", "Assumptions:", *notes]
    out.append("  The 'free' preset uses golden x1 + held-out x3 (the deterministic baseline always runs x3, offline).")
    return "\n".join(out)


def _human(seconds: float) -> str:
    if seconds < 90:
        return f"~{seconds:.0f} s"
    if seconds < 5400:
        return f"~{seconds / 60:.0f} min"
    return f"~{seconds / 3600:.1f} h"


# ============================================================================ CLI
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", help="comma-separated model ids from evals/models.toml (default: [benchmark].default_models)")
    p.add_argument("--budget", choices=sorted(PRESETS), default="full", help="full: golden x3 + held-out x3; free: golden x1 + held-out x3, paced to free-tier quotas")
    p.add_argument("--runs", type=int, help="runs per set (full budget only; minimum 3 - LLMs are not deterministic)")
    p.add_argument("--sets", default="golden,heldout", help="datasets to run (default: golden,heldout)")
    p.add_argument("--smoke", action="store_true", help="1 run on a few cases per set to measure calls/tokens per case; writes no result file")
    p.add_argument("--limit", type=int, help="cases per set for --smoke (default 5, spread across category groups)")
    p.add_argument("--parallel", action="store_true", help="run different providers concurrently (same provider stays sequential)")
    p.add_argument("--resume", nargs="?", const="latest", metavar="SESSION", help="continue a session (default: the latest)")
    p.add_argument("--allow-commit-change", action="store_true", help="with --resume: accept that the code changed since the session started")
    p.add_argument("--wait", action="store_true", help="when a daily quota is reached, sleep until it resets instead of exiting")
    p.add_argument("--estimate", action="store_true", help="print estimated cost and duration of both presets; no API call")
    p.add_argument("--summarize-only", action="store_true", help="rebuild SUMMARY.md from existing run files; no API call")
    p.add_argument("--trace", action="store_true", help="send LangSmith traces if LangSmith is configured (off by default here)")
    p.add_argument("--max-fallback-rate", type=float, help="override [benchmark].max_fallback_rate")
    return p.parse_args(argv)


def new_plan(args: argparse.Namespace, cfg: BenchmarkConfig, env: Settings, model_ids: list[str]) -> dict[str, Any]:
    sets = [s.strip() for s in args.sets.split(",") if s.strip()]
    bad = [s for s in sets if s not in DATASETS]
    if bad or not sets:
        raise BenchmarkError(f"--sets must be a subset of {','.join(DATASETS)}", EXIT_CONFIG)
    if args.limit is not None and not args.smoke:
        raise BenchmarkError("--limit is only allowed with --smoke (partial runs are never published)", EXIT_CONFIG)
    if args.smoke:
        runs = {ds: 1 for ds in sets}
    else:
        runs = {ds: PRESETS[args.budget][ds] for ds in sets}
        if args.runs is not None:
            if args.budget != "full":
                raise BenchmarkError("--runs only applies to --budget full", EXIT_CONFIG)
            if args.runs < 3:
                raise BenchmarkError("--runs must be at least 3: one LLM run says nothing about variance", EXIT_CONFIG)
            runs = {ds: args.runs for ds in sets}
    return {
        "budget": args.budget,
        "models": model_ids,
        "runs": runs,
        "smoke": bool(args.smoke),
        "limit": (args.limit or 5) if args.smoke else None,
        "git": git_info(),
        "dataset_hashes": {"golden": file_hash(GOLDEN_PATH), "heldout": file_hash(HELDOUT_PATH)},
        "llm_rpm": env.llm_rpm,
        "max_fallback_rate": args.max_fallback_rate if args.max_fallback_rate is not None else cfg.max_fallback_rate,
    }


async def main_async(args: argparse.Namespace, *, root: Path = ROOT, llm_factory: Callable[[ModelSpec, Settings], LLMClient | None] | None = None,
                     sleep: Callable[[float], Awaitable[None]] = asyncio.sleep) -> int:
    cfg = load_config()
    if args.summarize_only:
        print(f"summary: {write_summary(cfg, root)}")
        return 0
    env = env_settings()
    datasets = load_datasets()

    if args.resume:
        session = Session.load(root, args.resume)
        plan = session.plan
        if plan["dataset_hashes"] != {"golden": file_hash(GOLDEN_PATH), "heldout": file_hash(HELDOUT_PATH)}:
            raise BenchmarkError("--resume: a dataset changed since the session started; results would not be comparable", EXIT_CONFIG)
        current = git_info()
        if current["commit"] != plan["git"]["commit"] and not args.allow_commit_change:
            raise BenchmarkError(
                f"--resume: the code is at {current['commit'][:8]} but the session started at {plan['git']['commit'][:8]}. "
                "Results may mix agent versions; pass --allow-commit-change to continue anyway.",
                EXIT_CONFIG,
            )
        specs = [cfg.get(m) for m in plan["models"]]
    else:
        model_ids = [m.strip() for m in (args.models.split(",") if args.models else cfg.default_models) if m.strip()]
        specs = [cfg.get(m) for m in model_ids]
        if args.estimate:
            print(estimate(cfg, specs, env, datasets, root))
            return 0
        plan = new_plan(args, cfg, env, model_ids)

    check_keys(specs, env)  # before creating anything on disk
    if llm_factory is None:
        for spec in specs:
            if spec.is_llm:
                await check_model_available(spec, api_key(spec, env))
    if not args.resume:
        root.mkdir(parents=True, exist_ok=True)
        session = Session.create(root, plan)
    print(f"benchmark session {session.session_id} (budget={plan['budget']}, runs={plan['runs']}, models={plan['models']})", flush=True)

    ctx = Ctx(
        cfg=cfg, session=session, env=env, datasets=datasets, trace=args.trace, wait=args.wait, parallel=args.parallel,
        sleep=sleep, llm_factory=llm_factory,
    )
    try:
        await run_all(ctx, specs, parallel=args.parallel)
    except ProviderUnstable as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return EXIT_PREFLIGHT
    finally:
        if not plan.get("smoke"):
            print(f"summary: {write_summary(cfg, root)}")

    if plan.get("smoke"):
        print_smoke(ctx, specs)
    for written in ctx.written:
        print(f"run file: {written}")
    if ctx.paused:
        print("\nPaused (daily quota). Nothing was lost; resume with:")
        for p in ctx.paused:
            when = p.resume_after.astimezone().strftime("%Y-%m-%d %H:%M %Z") if p.resume_after else "later"
            print(f"  {p.model_id}: after {when}")
        print(f"  python -m evals.run_benchmark --resume {session.session_id}   (add --wait to keep it running)")
        return EXIT_PAUSED
    if ctx.invalid:
        print("\nINVALID runs (excluded from SUMMARY):\n  " + "\n  ".join(ctx.invalid), file=sys.stderr)
        return EXIT_INVALID
    return 0


def print_smoke(ctx: Ctx, specs: Sequence[ModelSpec]) -> None:
    print("\n=== smoke results (not published; feeds --estimate) ===")
    for spec in specs:
        for ds in ctx.session.plan["runs"]:
            records = list(ctx.session.completed(spec.id, ds, 1).values())
            if not records:
                continue
            s = summarize_set(records, spec)
            p = s["perf"]
            print(
                f"  {spec.id:<18}{ds:<8} pass {sum(r['passed'] for r in records)}/{len(records)}  "
                f"calls/case {p['llm_calls_per_case']}  tokens/case {p['input_tokens_per_case']} in / {p['output_tokens_per_case']} out  "
                f"cost/case ${p['cost_per_dispute_usd']}  fallbacks {p['degraded_cases']}"
            )
    print("Next: python -m evals.run_benchmark --estimate")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        code = asyncio.run(main_async(args))
    except BenchmarkError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        code = exc.code
    except KeyboardInterrupt:
        print("\ninterrupted: finished cases are saved; continue with --resume latest", file=sys.stderr)
        code = 130
    raise SystemExit(code)


if __name__ == "__main__":
    main()
