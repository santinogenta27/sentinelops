"""Benchmark sessions: an append-only event log that makes every run resumable.

    reports/benchmarks/.sessions/<session_id>/session.json   immutable plan (models, sets, runs, commit)
    reports/benchmarks/.sessions/<session_id>/events.jsonl   one line per finished/discarded case, probe, pause

A case is appended only once it has finished with a result attributable to the model. ``--resume``
replays the log and runs only what is missing, so an interrupted session (Ctrl-C, crash, daily quota
exhausted) never repeats or loses work. Every LLM request, including discarded cases and probes,
is logged with its token usage, which is how the free-tier daily budget is accounted for.
"""

from __future__ import annotations

import json
import secrets
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


def utc_now() -> datetime:
    return datetime.now(UTC)


def iso(ts: datetime) -> str:
    return ts.astimezone(UTC).isoformat(timespec="seconds")


def new_session_id(now: datetime | None = None) -> str:
    return f"{(now or utc_now()).strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}"


def quota_day_start(now: datetime, reset_utc_hour: int) -> datetime:
    """Start of the provider's current quota day (daily limits reset at ``reset_utc_hour`` UTC)."""
    start = now.astimezone(UTC).replace(hour=reset_utc_hour, minute=0, second=0, microsecond=0)
    return start if start <= now else start - timedelta(days=1)


def next_reset(now: datetime, reset_utc_hour: int) -> datetime:
    return quota_day_start(now, reset_utc_hour) + timedelta(days=1)


@dataclass(slots=True)
class Session:
    root: Path
    session_id: str
    plan: dict[str, Any]

    @property
    def dir(self) -> Path:
        return self.root / ".sessions" / self.session_id

    @property
    def events_path(self) -> Path:
        return self.dir / "events.jsonl"

    # ------------------------------------------------------------------ lifecycle
    @classmethod
    def create(cls, root: Path, plan: dict[str, Any], session_id: str | None = None) -> Session:
        sid = session_id or new_session_id()
        s = cls(root=root, session_id=sid, plan={**plan, "session_id": sid, "created_at": iso(utc_now())})
        s.dir.mkdir(parents=True, exist_ok=False)
        (s.dir / "session.json").write_text(json.dumps(s.plan, indent=2), encoding="utf-8")
        s.events_path.touch()
        return s

    @classmethod
    def load(cls, root: Path, session_id: str) -> Session:
        if session_id == "latest":
            sessions = sorted(p.name for p in (root / ".sessions").glob("*") if (p / "session.json").exists())
            if not sessions:
                raise SystemExit(f"--resume latest: no benchmark session found under {root / '.sessions'}")
            session_id = sessions[-1]
        path = root / ".sessions" / session_id / "session.json"
        if not path.exists():
            raise SystemExit(f"--resume: session {session_id!r} not found ({path})")
        return cls(root=root, session_id=session_id, plan=json.loads(path.read_text(encoding="utf-8")))

    # ------------------------------------------------------------------ events
    def append(self, event: dict[str, Any]) -> None:
        line = json.dumps({"ts": iso(utc_now()), **event}, default=str)
        with self.events_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def events(self) -> Iterator[dict[str, Any]]:
        yield from _read_events(self.events_path)

    def completed(self, model_id: str, dataset: str, run: int) -> dict[str, dict[str, Any]]:
        """case_id -> record for every case of (model, dataset, run) already finished in this session."""
        out: dict[str, dict[str, Any]] = {}
        for e in self.events():
            if e.get("type") == "case" and e["model"] == model_id and e["dataset"] == dataset and e["run"] == run:
                out[e["case_id"]] = e["record"]
        return out

    def run_written(self, model_id: str, run: int) -> bool:
        return any(e.get("type") == "run_written" and e["model"] == model_id and e["run"] == run for e in self.events())


def _read_events(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue  # a line truncated by a hard kill: the case simply runs again


def usage_since(root: Path, model_id: str, since: datetime) -> dict[str, int]:
    """LLM requests and tokens a model consumed since ``since``, across every session (quota accounting)."""
    calls = tokens = 0
    for path in (root / ".sessions").glob("*/events.jsonl"):
        for e in _read_events(path):
            if e.get("model") != model_id or "usage" not in e:
                continue
            if datetime.fromisoformat(e["ts"]) < since:
                continue
            u = e["usage"]
            calls += int(u.get("calls", 0))
            tokens += int(u.get("input_tokens", 0)) + int(u.get("output_tokens", 0))
    return {"calls": calls, "tokens": tokens}


def all_measurements(root: Path, model_id: str) -> dict[str, float] | None:
    """Average per-case usage of a model over every recorded session (feeds --estimate)."""
    perfs: list[dict[str, Any]] = []
    for path in (root / ".sessions").glob("*/events.jsonl"):
        perfs += [e["record"]["perf"] for e in _read_events(path) if e.get("type") == "case" and e.get("model") == model_id]
    if not perfs:
        return None
    n = len(perfs)
    return {
        "cases": n,
        "calls": sum(p["calls"] for p in perfs) / n,
        "input_tokens": sum(p["input_tokens"] for p in perfs) / n,
        "output_tokens": sum(p["output_tokens"] for p in perfs) / n,
        "effective_s": sum(p["effective_s"] for p in perfs) / n,
    }
