"""Render reports/benchmarks/SUMMARY.md from the raw per-run JSON files (never from anything else)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.config import BenchmarkConfig
from evals.stats import METRICS, aggregate_runs, flaky_cases

DATASETS = ("golden", "heldout")
LABEL = {"golden": "golden", "heldout": "held-out"}
METRIC_LABEL = {
    "case_pass_rate": "Case pass rate",
    "action_accuracy": "Action accuracy",
    "hitl_gate_accuracy": "HITL gate accuracy",
    "injection_defense_rate": "Injection defense",
    "hallucination_rate": "Hallucination rate ↓",
    "tool_precision": "Tool precision",
    "tool_recall": "Tool recall",
    "category_accuracy": "Category accuracy",
}


def load_runs(root: Path) -> list[dict[str, Any]]:
    runs = []
    for path in sorted(root.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(data, dict) and data.get("schema") == "sentinelops.benchmark.run/1":
            data["_file"] = path.name
            runs.append(data)
    return runs


def select_latest(runs: list[dict[str, Any]]) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Per model: the valid runs of its most recent session that has any valid run. Also returns invalid runs."""
    chosen: dict[str, list[dict[str, Any]]] = {}
    invalid = [r for r in runs if not r.get("valid")]
    by_model: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for r in runs:
        if r.get("valid"):
            by_model.setdefault(r["model"]["id"], {}).setdefault(r["session_id"], []).append(r)
    for model_id, sessions in by_model.items():
        latest = sorted(sessions)[-1]
        chosen[model_id] = sorted(sessions[latest], key=lambda r: int(r["run"]))
    return chosen, invalid


def _fmt(stat: dict[str, Any] | None, digits: int = 3) -> str:
    if not stat or stat.get("mean") is None:
        return "—"
    mean = f"{stat['mean']:.{digits}f}"
    if stat.get("sd") is None:
        return f"{mean} (n={stat['n']})"
    return f"{mean} ± {stat['sd']:.{digits}f}"


def _fmt_money(stat: dict[str, Any] | None) -> str:
    if not stat or stat.get("mean") is None:
        return "—"
    s = f"${stat['mean']:.5f}"
    return s if stat.get("sd") is None else f"{s} ± {stat['sd']:.5f}"


def render(cfg: BenchmarkConfig, root: Path) -> str:
    runs = load_runs(root)
    chosen, invalid = select_latest(runs)
    agg: dict[str, dict[str, dict[str, Any]]] = {}
    for model_id, mruns in chosen.items():
        agg[model_id] = {}
        for ds in DATASETS:
            sets = [r["sets"][ds] for r in mruns if ds in r["sets"]]
            if sets:
                agg[model_id][ds] = aggregate_runs(sets)

    model_ids = list(dict.fromkeys([*cfg.default_models, *cfg.models, *chosen]))
    lines: list[str] = [
        "# SentinelOps — LLM benchmark summary",
        "",
        f"_Generated {datetime.now(UTC).isoformat(timespec='seconds')} by `python -m evals.run_benchmark` from the raw "
        "per-run files in this folder. Do not edit by hand._",
        "",
        "**How to read it.** Each cell is `mean ± sample stdev` over the runs of the model's latest session "
        "(`(n=1)` = a single run, no variance measurable). LLMs are not deterministic, so a single run is not a result.",
        "The **held-out** set is never used to tune prompts or logic; it is only measured. Its headline numbers exclude "
        "contaminated cases: "
        + ", ".join(f"`{cid}` ({why})" for cid, why in cfg.contaminated_heldout.items())
        + ". The raw JSON keeps both views (`metrics` and `metrics_all_cases`).",
        "Runs where more than "
        f"{cfg.max_fallback_rate:.0%} of the cases fell back to the deterministic engine are invalid and excluded. "
        "Cases hit by provider quota/outage errors are discarded and re-run, never scored.",
        "",
        "## Models",
        "",
        "| Model | Provider / model id | Session | Runs (golden / held-out) | Budget | Commit | Prices (USD / 1M tok in · out) |",
        "|---|---|---|---|---|---|---|",
    ]
    for mid in model_ids:
        spec = cfg.models.get(mid)
        provider = f"{spec.provider} / `{spec.model}`" if spec else "?"
        price = (
            f"{spec.input_usd_per_mtok:.2f} · {spec.output_usd_per_mtok:.2f} (as of {spec.prices_as_of})" if spec else "?"
        )
        if mid not in chosen:
            lines.append(f"| {mid} | {provider} | **not run** | — | — | — | {price} |")
            continue
        r0 = chosen[mid][0]
        n_g = sum(1 for r in chosen[mid] if "golden" in r["sets"])
        n_h = sum(1 for r in chosen[mid] if "heldout" in r["sets"])
        commit = f"`{r0['git']['commit'][:8]}`" + (" (dirty)" if r0["git"].get("dirty") else "")
        lines.append(f"| {mid} | {provider} | `{r0['session_id']}` | {n_g} / {n_h} | {r0['budget']} | {commit} | {price} |")

    def quality_table(metrics: tuple[str, ...]) -> None:
        header = "| Model | " + " | ".join(f"{METRIC_LABEL[m]} — golden | {METRIC_LABEL[m]} — held-out" for m in metrics) + " |"
        lines.extend(["", header, "|---|" + "---|---|" * len(metrics)])
        for mid in model_ids:
            cells = []
            for m in metrics:
                for ds in DATASETS:
                    stat = agg.get(mid, {}).get(ds, {}).get("metrics", {}).get(m)
                    cells.append(_fmt(stat) if mid in chosen else "not run")
            lines.append(f"| {mid} | " + " | ".join(cells) + " |")

    lines += ["", "## Quality (golden | held-out)"]
    quality_table(METRICS[:4])
    quality_table(METRICS[4:])

    lines += [
        "",
        "## Latency, tokens and cost",
        "",
        "Latency is wall-clock per dispute (whole graph run incl. HITL resume). *Effective* subtracts time spent waiting "
        "on the client-side rate limiter, so it reflects the model, not `LLM_RPM`. Cost = measured tokens × the price "
        "table above; `*` = some responses carried no token counts, so cost is a lower bound.",
        "",
        "| Model | Set | p50 s | p95 s | eff. p50 s | eff. p95 s | LLM calls / dispute | input tok / dispute | "
        "output tok / dispute | cost / dispute | fallback rate |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for mid in model_ids:
        if mid not in chosen:
            continue
        for ds in DATASETS:
            a = agg[mid].get(ds)
            if not a:
                continue
            p = a["perf"]
            star = "" if p["cost_complete"] else "*"
            lines.append(
                f"| {mid} | {LABEL[ds]} | {_fmt(p['latency_p50_s'], 2)} | {_fmt(p['latency_p95_s'], 2)} | "
                f"{_fmt(p['effective_latency_p50_s'], 2)} | {_fmt(p['effective_latency_p95_s'], 2)} | "
                f"{_fmt(p['llm_calls_per_case'], 1)} | {_fmt(p['input_tokens_per_case'], 0)} | "
                f"{_fmt(p['output_tokens_per_case'], 0)} | {_fmt_money(p['cost_per_dispute_usd'])}{star} | "
                f"{_fmt(p['fallback_rate'])} |"
            )

    groups = sorted({g for mid in agg for ds in agg[mid] for g in agg[mid][ds]["groups"]})
    if groups:
        ran = [m for m in model_ids if m in chosen]
        lines += [
            "",
            "## Case pass rate by category group",
            "",
            "| Group | " + " | ".join(f"{m} — golden | {m} — held-out" for m in ran) + " |",
            "|---|" + "---|---|" * len(ran),
        ]
        for g in groups:
            cells = [_fmt(agg[m].get(ds, {}).get("groups", {}).get(g)) for m in ran for ds in DATASETS]
            lines.append(f"| {g} | " + " | ".join(cells) + " |")

    lines += ["", "## Unstable cases (outcome changed between runs)", ""]
    any_flaky = False
    for mid in model_ids:
        for ds in DATASETS:
            per_run = [r["sets"][ds]["results"] for r in chosen.get(mid, []) if ds in r["sets"]]
            for f in flaky_cases(per_run):
                any_flaky = True
                lines.append(f"- {mid} / {LABEL[ds]}: `{f['case_id']}` passed {f['passed_runs']}/{f['runs']} runs")
    if not any_flaky:
        lines.append("- none")

    lines += ["", "## Invalid runs (excluded above)", ""]
    lines += [f"- `{r['_file']}`: {r.get('invalid_reason')}" for r in invalid] or ["- none"]

    notes = [(mid, cfg.models[mid]) for mid in model_ids if mid in cfg.models and cfg.models[mid].notes]
    if notes:
        lines += ["", "## Model notes", ""]
        for mid, spec in notes:
            extra = f" Price after {spec.price_valid_until}: {spec.price_after}." if spec.price_after else ""
            lines.append(f"- **{mid}**: {spec.notes.strip()}{extra}")
    return "\n".join(lines) + "\n"


def write_summary(cfg: BenchmarkConfig, root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "SUMMARY.md"
    path.write_text(render(cfg, root), encoding="utf-8")
    return path
