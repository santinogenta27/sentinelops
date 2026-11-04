"""Pure aggregation helpers: per-set metrics, latency percentiles, tokens, cost and run-to-run variation."""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import fields
from typing import Any

from evals.config import ModelSpec
from tests.eval_suite import CaseResult, _metrics, _metrics_by_group

_CASE_FIELDS = {f.name for f in fields(CaseResult)}

# Order used in reports. All come from tests/eval_suite.py::_metrics.
METRICS: tuple[str, ...] = (
    "case_pass_rate",
    "action_accuracy",
    "hitl_gate_accuracy",
    "injection_defense_rate",
    "hallucination_rate",
    "tool_precision",
    "tool_recall",
    "category_accuracy",
)
LOWER_IS_BETTER = {"hallucination_rate"}


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolated percentile (q in [0, 100]); None for an empty sample."""
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    k = (len(xs) - 1) * q / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    return float(xs[lo] + (xs[hi] - xs[lo]) * (k - lo))


def mean_sd(values: Sequence[float]) -> dict[str, float | int | None]:
    """Mean and sample standard deviation (None with fewer than two runs: no variance can be measured)."""
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "sd": None, "min": None, "max": None}
    return {
        "n": n,
        "mean": round(statistics.fmean(values), 4),
        "sd": round(statistics.stdev(values), 4) if n >= 2 else None,
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def case_result(record: dict[str, Any]) -> CaseResult:
    return CaseResult(**{k: v for k, v in record.items() if k in _CASE_FIELDS})


def perf_summary(records: Sequence[dict[str, Any]], spec: ModelSpec) -> dict[str, Any]:
    perf = [r["perf"] for r in records]
    n = len(perf)
    wall = [p["wall_s"] for p in perf]
    eff = [p["effective_s"] for p in perf]
    tin = sum(p["input_tokens"] for p in perf)
    tout = sum(p["output_tokens"] for p in perf)
    calls = sum(p["calls"] for p in perf)
    missing = sum(p["calls_without_usage"] for p in perf)
    degraded = sum(1 for r in records if r.get("llm_degraded"))
    cost = spec.cost_usd(tin, tout)

    def r4(x: float | None) -> float | None:
        return None if x is None else round(x, 4)

    return {
        "cases": n,
        "latency_p50_s": r4(percentile(wall, 50)),
        "latency_p95_s": r4(percentile(wall, 95)),
        # wall time minus the time spent waiting on the client-side rate limiter (LLM_RPM / quota pacing)
        "effective_latency_p50_s": r4(percentile(eff, 50)),
        "effective_latency_p95_s": r4(percentile(eff, 95)),
        "llm_calls": calls,
        "llm_calls_per_case": r4(calls / n) if n else None,
        "retries": sum(p["retries"] for p in perf),
        "rate_limited_attempts": sum(p["rate_limited_attempts"] for p in perf),
        "input_tokens": tin,
        "output_tokens": tout,
        "input_tokens_per_case": r4(tin / n) if n else None,
        "output_tokens_per_case": r4(tout / n) if n else None,
        "calls_without_usage": missing,
        "cost_usd": round(cost, 6),
        "cost_per_dispute_usd": round(cost / n, 6) if n else None,
        # cost is a lower bound when some responses carried no token counts
        "cost_complete": missing == 0,
        "degraded_cases": degraded,
        "fallback_rate": r4(degraded / n) if n else None,
    }


def summarize_set(records: Sequence[dict[str, Any]], spec: ModelSpec, contaminated: Iterable[str] = ()) -> dict[str, Any]:
    """Metrics for one model x dataset x run. ``contaminated`` ids are excluded from the headline metrics."""
    excluded = set(contaminated)
    results = [case_result(r) for r in records]
    clean = [r for r in results if r.case_id not in excluded]
    out: dict[str, Any] = {
        "cases": len(results),
        "metrics": _metrics(clean) if clean else {},
        "metrics_by_group": _metrics_by_group(clean) if clean else {},
        "perf": perf_summary(records, spec),
        "failed_cases": [r.case_id for r in clean if not r.passed],
    }
    if excluded & {r.case_id for r in results}:
        out["headline_excludes"] = sorted(excluded & {r.case_id for r in results})
        out["clean_cases"] = len(clean)
        out["metrics_all_cases"] = _metrics(results)
    return out


def aggregate_runs(set_summaries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Mean/sd/min/max across runs of every headline metric, perf figure and per-group pass rate."""
    out: dict[str, Any] = {"runs": len(set_summaries), "metrics": {}, "perf": {}, "groups": {}}
    for m in METRICS:
        vals = [s["metrics"][m] for s in set_summaries if m in s.get("metrics", {})]
        out["metrics"][m] = mean_sd(vals)
    for key in (
        "latency_p50_s",
        "latency_p95_s",
        "effective_latency_p50_s",
        "effective_latency_p95_s",
        "llm_calls_per_case",
        "input_tokens_per_case",
        "output_tokens_per_case",
        "cost_per_dispute_usd",
        "cost_usd",
        "fallback_rate",
    ):
        vals = [s["perf"][key] for s in set_summaries if s["perf"].get(key) is not None]
        out["perf"][key] = mean_sd(vals)
    out["perf"]["cost_complete"] = all(s["perf"]["cost_complete"] for s in set_summaries)
    groups = sorted({g for s in set_summaries for g in s.get("metrics_by_group", {})})
    for g in groups:
        vals = [s["metrics_by_group"][g]["case_pass_rate"] for s in set_summaries if g in s.get("metrics_by_group", {})]
        out["groups"][g] = mean_sd(vals)
    return out


def flaky_cases(per_run_records: Sequence[Sequence[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Cases whose pass/fail outcome differs between runs of the same model and dataset."""
    outcomes: dict[str, list[bool]] = {}
    for records in per_run_records:
        for r in records:
            outcomes.setdefault(r["case_id"], []).append(bool(r["passed"]))
    return [
        {"case_id": cid, "passed_runs": sum(v), "runs": len(v)}
        for cid, v in sorted(outcomes.items())
        if len(v) > 1 and 0 < sum(v) < len(v)
    ]
