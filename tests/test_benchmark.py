"""Benchmark harness (evals/): honesty rules, aggregation, quota pacing and resume. No network, no API key."""

from __future__ import annotations

import json
from argparse import Namespace
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from app.config import Settings
from app.llm.base import LLMClient, LLMError, LLMTask
from evals import run_benchmark as rb
from evals.config import BenchmarkConfig, FreeTier, ModelSpec, load_config
from evals.session import Session, iso, quota_day_start, utc_now
from evals.stats import flaky_cases, mean_sd, percentile, summarize_set
from tests.eval_suite import CASES as GOLDEN

HELDOUT: list[dict[str, Any]] = json.loads(rb.HELDOUT_PATH.read_text(encoding="utf-8"))
SMALL = {
    "golden": rb.stratified_sample(GOLDEN, 3),  # one case from each of the first three category groups
    "heldout": [c for c in HELDOUT if c["id"] in {"ho_inj_fake_system_json", "ho_unk_unregistered_email"}],
}


# ------------------------------------------------------------------ fakes
class FakeLLM(LLMClient):
    """Answers every request with unusable text (the agent falls back = model failure), or fails like a provider."""

    provider = "groq"

    def __init__(self, *, fail_from_call: int | None = None, daily: bool = False) -> None:
        super().__init__(model="fake-model", rpm=100_000, max_retries=0, timeout_s=5)
        self.fail_from_call = fail_from_call
        self.daily = daily
        self.n = 0

    async def _complete(self, *, system: str, prompt: str, json_mode: bool) -> str:
        self.n += 1
        if self.fail_from_call is not None and self.n >= self.fail_from_call:
            scope = "requests per day (RPD)" if self.daily else "requests per minute (RPM)"
            raise LLMError(f"groq HTTP 429: Rate limit reached on {scope}", retryable=True, status_code=429)
        self._record_tokens(1000, 50)
        return "this is not json"


FAKE_SPEC = ModelSpec(
    id="fake-groq",
    provider="groq",
    model="fake-model",
    api_key_env="GROQ_API_KEY",
    input_usd_per_mtok=1.0,
    output_usd_per_mtok=2.0,
    prices_as_of="2026-11-04",
    price_source="test",
)


@pytest.fixture
def bench(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Small datasets, a config with the fake model and env settings that never read the developer's .env."""
    cfg = load_config()
    cfg.models[FAKE_SPEC.id] = FAKE_SPEC
    monkeypatch.setattr(rb, "load_config", lambda: cfg)
    monkeypatch.setattr(rb, "load_datasets", lambda: {k: list(v) for k, v in SMALL.items()})
    monkeypatch.setattr(rb, "env_settings", lambda: Settings(_env_file=None, groq_api_key="test-key", gemini_api_key=None))  # type: ignore[call-arg]
    return tmp_path / "benchmarks"


def args(**kw: Any) -> Namespace:
    base: dict[str, Any] = {
        "models": "deterministic",
        "budget": "full",
        "runs": None,
        "sets": "golden,heldout",
        "smoke": False,
        "limit": None,
        "parallel": False,
        "resume": None,
        "allow_commit_change": True,
        "wait": False,
        "estimate": False,
        "summarize_only": False,
        "trace": False,
        "max_fallback_rate": None,
    }
    base.update(kw)
    return Namespace(**base)


async def no_sleep(_: float) -> None:
    return None


def factory(llm: LLMClient) -> Callable[[ModelSpec, Settings], LLMClient | None]:
    return lambda spec, settings: llm


def case_events(root: Path, kind: str = "case") -> list[dict[str, Any]]:
    out = []
    for p in (root / ".sessions").glob("*/events.jsonl"):
        out += [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [e for e in out if e.get("type") == kind]


# ------------------------------------------------------------------ pure helpers
def test_percentile_and_variation() -> None:
    assert percentile([], 50) is None
    assert percentile([4.0], 95) == 4.0
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95) == pytest.approx(9.55)
    one = mean_sd([0.9])
    assert one["mean"] == 0.9 and one["sd"] is None  # a single run has no measurable variance
    three = mean_sd([0.9, 1.0, 0.8])
    assert three["mean"] == pytest.approx(0.9) and three["sd"] == pytest.approx(0.1)


def test_cost_and_flaky_cases() -> None:
    spec = ModelSpec(
        id="m", provider="groq", model="m", api_key_env="K", input_usd_per_mtok=0.15, output_usd_per_mtok=0.60,
        prices_as_of="2026-11-04", price_source="t",
    )
    assert spec.cost_usd(1_000_000, 500_000) == pytest.approx(0.45)
    runs = [[{"case_id": "a", "passed": True}, {"case_id": "b", "passed": True}], [{"case_id": "a", "passed": False}, {"case_id": "b", "passed": True}]]
    assert flaky_cases(runs) == [{"case_id": "a", "passed_runs": 1, "runs": 2}]


def test_models_file_is_consistent() -> None:
    cfg = load_config()
    assert set(cfg.default_models) <= set(cfg.models)
    assert cfg.models["deterministic"].provider == "mock"
    for spec in cfg.models.values():
        assert spec.prices_as_of, f"{spec.id} needs the date of its prices"
        if spec.is_llm:
            assert spec.api_key_env and spec.free_tier is not None
    assert "ho_inj_fake_system_json" in cfg.contaminated_heldout


async def test_llm_usage_classifies_model_vs_provider_failures() -> None:
    llm = FakeLLM()
    with pytest.raises(LLMError):
        await llm.generate_structured(task=LLMTask.TRIAGE, system="s", prompt="p", schema=rb_schema())
    u = llm.usage
    assert u.calls == 3 and u.input_tokens == 3000 and u.output_tokens == 150  # 1 try + 2 self-repairs
    assert u.surfaced_model_errors == 1 and u.surfaced_infra_errors == 0

    flaky = FakeLLM(fail_from_call=1)
    with pytest.raises(LLMError):
        await flaky.generate_text(task=LLMTask.TRIAGE, system="s", prompt="p")
    assert flaky.usage.surfaced_infra_errors == 1 and flaky.usage.rate_limited_attempts == 1
    assert flaky.usage.input_tokens == 0 and flaky.usage.calls_without_usage == 0


def rb_schema() -> Any:
    from app.models.schemas import JudgeVerdict

    return JudgeVerdict


async def test_budget_limiter_waits_when_the_token_window_is_full() -> None:
    class Waited(Exception):
        pass

    async def sleep(_: float) -> None:
        raise Waited

    limiter = rb.BudgetLimiter(rpm=1000, tpm=1500, sleep=sleep)
    await limiter.acquire()  # empty window: no wait
    limiter.on_tokens(1000)
    with pytest.raises(Waited):
        await limiter.acquire()  # 1000 used + ~1000 expected > 1500 TPM


def test_daily_budget_check_pauses_before_overrunning(tmp_path: Path) -> None:
    spec = ModelSpec(
        id="free-model", provider="groq", model="x", api_key_env="K", input_usd_per_mtok=0, output_usd_per_mtok=0,
        prices_as_of="2026-11-04", price_source="t",
        free_tier=FreeTier(rpm=30, rpd=20, tpm=0, tpd=0, daily_reset_utc_hour=0, quota_source="t"),
    )
    session = Session.create(tmp_path, {"budget": "free", "runs": {"golden": 1}})
    ctx = rb.Ctx(cfg=load_config(), session=session, env=Settings(_env_file=None), datasets={})  # type: ignore[call-arg]
    rb.check_daily_budget(ctx, spec)  # nothing used yet
    session.append({"type": "discard", "model": spec.id, "usage": {"calls": 15, "input_tokens": 0, "output_tokens": 0}})
    with pytest.raises(rb.QuotaPaused) as info:
        rb.check_daily_budget(ctx, spec)  # 15 + 4.5 > 0.9 * 20
    assert info.value.resume_after is not None and info.value.resume_after > utc_now()
    assert quota_day_start(utc_now(), 0) <= utc_now()
    assert iso(utc_now()).endswith("+00:00")


# ------------------------------------------------------------------ end-to-end (deterministic engine + fakes)
async def test_missing_api_key_fails_before_anything_is_written(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(rb, "env_settings", lambda: Settings(_env_file=None, groq_api_key=None, gemini_api_key=None))  # type: ignore[call-arg]
    root = tmp_path / "benchmarks"
    with pytest.raises(rb.BenchmarkError) as info:
        await rb.main_async(args(models="deterministic,gpt-oss-120b"), root=root)
    assert info.value.code == rb.EXIT_CONFIG
    assert "GROQ_API_KEY is not set" in str(info.value)
    assert not root.exists()


async def test_deterministic_baseline_writes_runs_and_summary(bench: Path) -> None:
    code = await rb.main_async(args(), root=bench, sleep=no_sleep)
    assert code == 0
    runs = sorted(bench.glob("*_deterministic_r*.json"))
    assert len(runs) == 3
    doc = json.loads(runs[0].read_text(encoding="utf-8"))
    assert doc["valid"] and doc["model"]["prices_as_of"]
    held = doc["sets"]["heldout"]
    assert held["headline_excludes"] == ["ho_inj_fake_system_json"] and held["clean_cases"] == len(SMALL["heldout"]) - 1
    summary = (bench / "SUMMARY.md").read_text(encoding="utf-8")
    assert "| deterministic |" in summary
    assert "± 0.000" in summary  # three identical deterministic runs
    assert "**not run**" in summary  # LLM models without results get no numbers


async def test_model_fallbacks_make_runs_invalid(bench: Path) -> None:
    code = await rb.main_async(args(models="fake-groq"), root=bench, llm_factory=factory(FakeLLM()), sleep=no_sleep)
    assert code == rb.EXIT_INVALID
    docs = [json.loads(p.read_text(encoding="utf-8")) for p in bench.glob("*_fake-groq_r*.json")]
    assert len(docs) == 3 and not any(d["valid"] for d in docs)
    perf = docs[0]["sets"]["golden"]["perf"]
    assert perf["fallback_rate"] == 1.0 and perf["input_tokens"] > 0 and perf["cost_usd"] > 0
    summary = (bench / "SUMMARY.md").read_text(encoding="utf-8")
    assert "fake-groq | groq / `fake-model` | **not run**" in summary
    assert "fallback rate 100.0%" in summary


async def test_provider_errors_discard_cases_instead_of_scoring_them(bench: Path) -> None:
    llm = FakeLLM(fail_from_call=2)  # the probe succeeds, then every request is rate limited
    code = await rb.main_async(args(models="fake-groq", smoke=True, limit=1, sets="golden"), root=bench, llm_factory=factory(llm), sleep=no_sleep)
    assert code == rb.EXIT_PREFLIGHT
    assert case_events(bench) == []  # nothing was scored
    discards = case_events(bench, "discard")
    assert len(discards) == rb.MAX_CONSECUTIVE_DISCARDS
    # after the breaker opens, the agent skips the LLM without raising: those cases must be discarded too
    assert any("circuit breaker open" in d["reason"] for d in discards)


async def test_daily_quota_pauses_and_resume_completes_without_duplicates(bench: Path) -> None:
    first = await rb.main_async(
        args(models="fake-groq", max_fallback_rate=1.0), root=bench, llm_factory=factory(FakeLLM(fail_from_call=40, daily=True)), sleep=no_sleep
    )
    assert first == rb.EXIT_PAUSED
    done_before = len(case_events(bench))
    assert 0 < done_before < 15
    assert case_events(bench, "pause")

    second = await rb.main_async(args(resume="latest", max_fallback_rate=1.0), root=bench, llm_factory=factory(FakeLLM()), sleep=no_sleep)
    assert second == 0
    events = case_events(bench)
    keys = [(e["dataset"], e["run"], e["case_id"]) for e in events]
    assert len(keys) == len(set(keys)) == 3 * len(SMALL["golden"]) + 3 * len(SMALL["heldout"])
    assert len(list(bench.glob("*_fake-groq_r*.json"))) == 3


async def test_estimate_calls_no_api(bench: Path) -> None:
    code = await rb.main_async(args(models="deterministic,fake-groq", estimate=True), root=bench)
    assert code == 0 and not bench.exists()


def test_cli_guards() -> None:
    cfg: BenchmarkConfig = load_config()
    env = Settings(_env_file=None)  # type: ignore[call-arg]
    with pytest.raises(rb.BenchmarkError, match="at least 3"):
        rb.new_plan(args(runs=2), cfg, env, ["deterministic"])
    with pytest.raises(rb.BenchmarkError, match="only allowed with --smoke"):
        rb.new_plan(args(limit=3), cfg, env, ["deterministic"])
    free = rb.new_plan(args(budget="free"), cfg, env, ["deterministic"])
    assert free["runs"] == {"golden": 1, "heldout": 3}
