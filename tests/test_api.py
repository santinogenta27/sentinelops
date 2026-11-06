"""End-to-end API tests through FastAPI (lifespan, background runs, HITL decision endpoint)."""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    from app.main import app

    with TestClient(app) as c:
        yield c
    get_settings.cache_clear()


def wait_for(client: TestClient, tid: str, statuses: set[str], timeout: float = 20) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body: dict[str, Any] = client.get(f"/api/tickets/{tid}").json()
        status = body.get("execution_status")
        paused = status != "AWAITING_APPROVAL" or body.get("pending_interrupt") is not None
        if status in statuses and paused:
            return body
        time.sleep(0.1)
    raise AssertionError(f"{tid} never reached {statuses}")


def test_health_and_topology(client: TestClient) -> None:
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and h["llm"]["provider"] == "deterministic"
    g = client.get("/api/graph").json()
    assert "human_gate" in g["nodes"] and any(e["source"] == e["target"] == "db_inspector" for e in g["edges"])
    assert client.get("/").status_code == 200


def test_validation_rejects_bad_payloads(client: TestClient) -> None:
    assert client.post("/api/tickets", json={"message": "short"}).status_code == 422
    assert client.post("/api/tickets", json={"message": "valid message text", "customer_email": "nope"}).status_code == 422
    assert client.post("/api/tickets", json={"message": "valid message text", "admin": True}).status_code == 422


def test_full_hitl_flow_over_http(client: TestClient) -> None:
    r = client.post("/api/tickets", json={"message": "We were double charged $480 for the annual plan", "customer_email": "bob@startup.io"})
    assert r.status_code == 202
    tid = r.json()["ticket_id"]
    t = wait_for(client, tid, {"AWAITING_APPROVAL"})
    assert t["pending_interrupt"]["amount"] == 480.0
    assert [a["ticket_id"] for a in client.get("/api/approvals").json()] == [tid]

    # rejection without reason is invalid
    assert client.post(f"/api/tickets/{tid}/decision", json={"approved": False, "reviewer": "sup.a"}).status_code == 422
    assert client.post(f"/api/tickets/{tid}/decision", json={"approved": True, "reviewer": "sup.a"}).status_code == 202
    t = wait_for(client, tid, {"EXECUTED"})
    assert t["execution_receipt"]["refunds"][0]["amount"] == 480.0
    assert client.post(f"/api/tickets/{tid}/decision", json={"approved": True, "reviewer": "sup.b"}).status_code == 409

    ledger = client.get("/api/ledger").json()
    assert [r["ticket_id"] for r in ledger["refunds"]] == [tid]
    assert ledger["outbox"][0]["ticket_id"] == tid
    types = [e["type"] for e in client.get(f"/api/tickets/{tid}/events").json()]
    assert types[0] == "ticket_created" and "interrupt" in types and "decision" in types and types[-1] == "run_completed"


def test_fault_injection_endpoint_drives_self_correction(client: TestClient) -> None:
    assert client.post("/api/admin/faults", json={"tool": "db.query_transactions", "times": 2, "kind": "timeout"}).status_code == 200
    tid = client.post("/api/tickets", json={"message": "Refund my $29 Pro Monthly charge please.", "customer_email": "erin@designhub.co"}).json()["ticket_id"]
    t = wait_for(client, tid, {"EXECUTED", "ESCALATED", "FAILED"})
    assert t["execution_status"] == "EXECUTED" and t["tool_error_count"] == 2


def test_unknown_ticket_404(client: TestClient) -> None:
    assert client.get("/api/tickets/TCK-NOPE0000").status_code == 404
    assert client.post("/api/tickets/TCK-NOPE0000/decision", json={"approved": True, "reviewer": "sup"}).status_code == 404
