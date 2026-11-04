"""Held-out evaluation: measures whether agent fixes generalise beyond the golden set.

    python -m tests.heldout_eval --check-only   # validate the dataset only, never runs the agent
    python -m tests.heldout_eval                # single run -> reports/heldout_report.json

The held-out set (tests/datasets/heldout_tickets.json) mirrors the golden set's group distribution
with different wording, amounts, languages and customers. Its expected outcomes are derived from the
policy handbook and the seeded ledger only. It is deliberately NOT collected by pytest and NOT part
of the CI gate: it must never be used to tune the agent, or it stops being held out.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Same hermetic environment as tests/conftest.py (this script runs outside pytest, so conftest is not
# loaded): never reach the real Stripe account, and keep Auto-Pilot OFF so the harness — not the
# runtime — takes every HITL decision (otherwise "reject" cases get auto-approved).
os.environ["STRIPE_API_KEY"] = ""
os.environ["AUTOPILOT_DEFAULT"] = "false"
os.environ["ENABLE_EMAIL_LISTENER"] = "false"
os.environ["ENABLE_EMAIL_SENDER"] = "false"

from app.config import PROJECT_ROOT  # noqa: E402
from tests.eval_suite import (  # noqa: E402
    CASES as GOLDEN_CASES,
    LIVE,
    THRESHOLDS,
    _metrics,
    _metrics_by_group,
    _print_group_table,
    _run_all,
    _SEED_EMAILS,
    _SEED_TXNS,
    check_dataset,
)
from app.db.seed import EVAL_CUSTOMERS, HELDOUT_CUSTOMERS  # noqa: E402

DATASET = Path(__file__).parent / "datasets" / "heldout_tickets.json"
REPORT = PROJECT_ROOT / "reports" / "heldout_report.json"
_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,24}")
_TXN = re.compile(r"\btxn_[A-Za-z0-9]{4,32}\b")


def preflight(cases: list[dict[str, Any]]) -> None:
    """Dataset checks that never look at agent output."""
    check_dataset(cases)
    golden_ids = {c["id"] for c in GOLDEN_CASES}
    golden_msgs = {c["message"].strip().lower() for c in GOLDEN_CASES}
    assert not {c["id"] for c in cases} & golden_ids, "held-out ids overlap the golden set"
    assert not {c["message"].strip().lower() for c in cases} & golden_msgs, "held-out messages copied from the golden set"
    assert not {c.email for c in HELDOUT_CUSTOMERS} & {c.email for c in EVAL_CUSTOMERS}, "held-out customers reused"
    for c in cases:
        if c["category_group"] == "unknown_customer":
            continue
        emails = {e.lower() for e in _EMAIL.findall(c["message"])} | ({c["customer_email"]} if c.get("customer_email") else set())
        txns = set(_TXN.findall(c["message"]))
        assert emails & _SEED_EMAILS or txns & _SEED_TXNS.keys(), f"{c['id']}: customer cannot be identified from the seed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check-only", action="store_true", help="validate the dataset without running the agent")
    args = parser.parse_args()

    cases: list[dict[str, Any]] = json.loads(DATASET.read_text(encoding="utf-8"))
    preflight(cases)
    print(f"held-out dataset OK: {len(cases)} cases")
    if args.check_only:
        return

    results, provider = asyncio.run(_run_all(cases, heldout=True))
    metrics = _metrics(results)
    by_group = _metrics_by_group(results)
    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset": "heldout",
        "mode": "live" if LIVE else "deterministic",
        "provider": provider,
        "cases": len(results),
        "thresholds_for_reference": THRESHOLDS,
        "metrics": metrics,
        "metrics_by_group": by_group,
        "results": [asdict(r) for r in results],
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n=== SentinelOps held-out eval ===")
    for k, v in metrics.items():
        print(f"  {k:<24} {v:.3f}")
    _print_group_table(results, by_group)
    print(f"\nreport: {REPORT}")


if __name__ == "__main__":
    main()
