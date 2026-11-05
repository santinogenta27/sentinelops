"""DEMO_MODE: with it on, nothing leaves the server; with it off, the app behaves exactly as before."""

from __future__ import annotations

import asyncio
import socket
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.config import DEMO_BANNER, Settings, get_settings
from app.db.database import Database
from app.demo import SlidingWindowLimiter, client_ip
from app.scenarios import DEMO_SCENARIOS, SCENARIOS
from app.services.mailbox import MAILBOX_KEY

RESTING = {"EXECUTED", "REJECTED", "ESCALATED", "FAILED", "AWAITING_APPROVAL"}

# Every external integration configured with realistic-looking credentials: demo mode must ignore all of them.
REAL_LOOKING_ENV = {
    "LLM_PROVIDER": "gemini",
    "GEMINI_API_KEY": "AIza-not-a-real-key",
    "GROQ_API_KEY": "gsk-not-a-real-key",
    "EMBEDDING_PROVIDER": "gemini",
    "STRIPE_API_KEY": "sk_test_not_a_real_key",
    "STRIPE_MIRROR_REFUNDS": "true",
    "ENABLE_EMAIL_LISTENER": "true",
    "ENABLE_EMAIL_SENDER": "true",
    "IMAP_USER": "support@example.com",
    "IMAP_PASSWORD": "app-password",
    "SMTP_USER": "support@example.com",
    "SMTP_PASSWORD": "app-password",
    "SUPERVISOR_EMAIL": "boss@example.com",  # not *.demo: a real alert would be emailed
    "LANGSMITH_TRACING": "true",
    "LANGSMITH_API_KEY": "lsv2-not-a-real-key",
}


# ------------------------------------------------------------------ settings
def test_demo_settings_switch_off_every_external_integration() -> None:
    s = Settings(_env_file=None, demo_mode=True, **{k.lower(): v for k, v in REAL_LOOKING_ENV.items()})  # type: ignore[arg-type]
    assert s.llm_provider == "mock" and s.gemini_api_key is None and s.groq_api_key is None
    assert s.embedding_provider == "local"
    assert s.stripe_key is None and s.stripe_mirror_refunds is False
    assert s.enable_email_listener is False and s.enable_email_sender is False and s.smtp_credentials is None
    assert s.imap_password is None and s.langsmith_tracing is False and s.langsmith_api_key is None


def test_settings_without_demo_mode_are_untouched() -> None:
    s = Settings(_env_file=None, **{k.lower(): v for k, v in REAL_LOOKING_ENV.items()})  # type: ignore[arg-type]
    assert s.demo_mode is False
    assert s.llm_provider == "gemini" and s.gemini_api_key is not None and s.embedding_provider == "gemini"
    assert s.stripe_key == "sk_test_not_a_real_key" and s.stripe_mirror_refunds is True
    assert s.enable_email_listener is True and s.enable_email_sender is True
    assert s.smtp_credentials == ("support@example.com", "app-password")
    assert s.langsmith_tracing is True


# ------------------------------------------------------------------ network guard
class OutboundGuard:
    """Fails any connection or DNS lookup that is not loopback (asyncio's self-pipe on Windows uses loopback)."""

    LOCAL = {"127.0.0.1", "::1", "localhost", "0.0.0.0", "", "None"}

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.attempts: list[str] = []
        guard = self

        def _host(address: Any) -> str:
            return str(address[0]) if isinstance(address, tuple) else str(address)

        def check(host: Any, what: str) -> None:
            h = _host(host)
            if h not in self.LOCAL and not h.startswith("127.") and "/" not in h and "\\" not in h:
                guard.attempts.append(f"{what} {h}")
                raise OSError(f"blocked outbound {what} to {h} (demo-mode test)")

        real_connect, real_connect_ex = socket.socket.connect, socket.socket.connect_ex
        real_create, real_gai = socket.create_connection, socket.getaddrinfo
        real_loop_create = asyncio.base_events.BaseEventLoop.create_connection
        real_loop_sock_connect = asyncio.base_events.BaseEventLoop.sock_connect

        def connect(sock: socket.socket, address: Any) -> None:
            check(address, "connect")
            real_connect(sock, address)

        def connect_ex(sock: socket.socket, address: Any) -> int:
            check(address, "connect")
            return real_connect_ex(sock, address)

        def create_connection(address: Any, *a: Any, **kw: Any) -> socket.socket:
            check(address, "connect")
            return real_create(address, *a, **kw)

        def getaddrinfo(host: Any, *a: Any, **kw: Any) -> Any:
            check(host, "dns")
            return real_gai(host, *a, **kw)

        async def loop_create(loop: Any, protocol_factory: Any, host: Any = None, *a: Any, **kw: Any) -> Any:
            check(host, "connect")
            return await real_loop_create(loop, protocol_factory, host, *a, **kw)

        async def loop_sock_connect(loop: Any, sock: Any, address: Any) -> Any:
            check(address, "connect")
            return await real_loop_sock_connect(loop, sock, address)

        monkeypatch.setattr(socket.socket, "connect", connect)
        monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
        monkeypatch.setattr(socket, "create_connection", create_connection)
        monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
        monkeypatch.setattr(asyncio.base_events.BaseEventLoop, "create_connection", loop_create)
        monkeypatch.setattr(asyncio.base_events.BaseEventLoop, "sock_connect", loop_sock_connect)


def make_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, env: dict[str, str]) -> TestClient:
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    from app.main import app

    return TestClient(app)


@pytest.fixture(autouse=True)
def _fresh_settings() -> Iterator[None]:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def wait_rest(client: TestClient, tid: str, timeout: float = 60, *, final: bool = False) -> dict[str, Any]:
    targets = RESTING - {"AWAITING_APPROVAL"} if final else RESTING
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        t: dict[str, Any] = client.get(f"/api/tickets/{tid}").json()
        status = t.get("execution_status")
        if status in targets and (status != "AWAITING_APPROVAL" or t.get("pending_interrupt") is not None):
            return t
        time.sleep(0.1)
    raise AssertionError(f"{tid} never came to rest")


def run_to_end(client: TestClient, tid: str) -> dict[str, Any]:
    t = wait_rest(client, tid)
    if t["execution_status"] == "AWAITING_APPROVAL":
        r = client.post(f"/api/tickets/{tid}/decision", json={"approved": True, "reviewer": "demo.visitor"})
        assert r.status_code == 202
        t = wait_rest(client, tid, final=True)
    return t


async def _seed_saved_mailbox(data_dir: Path) -> None:
    """A real mailbox connected from the cockpit in an earlier (non-demo) run: demo mode must not reconnect it."""
    db = Database(data_dir / "sentinelops.db")
    await db.init_schema()
    await db.set_setting(MAILBOX_KEY, {
        "connected": True, "provider": "gmail", "email": "support@example.com", "app_password": "app-password",
        "imap_host": "imap.example.com", "imap_port": 993, "smtp_host": "smtp.example.com", "smtp_port": 587,
        "folder": "INBOX", "poll_interval_s": 5, "enable_sender": True, "keywords": None,
    })


# ------------------------------------------------------------------ DEMO_MODE=true
def test_demo_mode_nothing_leaves_the_server(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    asyncio.run(_seed_saved_mailbox(tmp_path))
    guard = OutboundGuard(monkeypatch)
    env = {**REAL_LOOKING_ENV, "DEMO_MODE": "true", "DEMO_WRITE_LIMIT_PER_MINUTE": "1000", "DEMO_RATE_LIMIT_PER_MINUTE": "100000"}
    with make_client(monkeypatch, tmp_path, env) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/api/demo").json()["banner"] == DEMO_BANNER == "Demo mode: simulated payments and email"
        h = client.get("/api/health").json()
        assert h["llm"]["provider"] == "deterministic" and h["embedder"] != "gemini"
        assert h["email"]["listener_enabled"] is False and h["email"]["sender_enabled"] is False
        assert h["stripe"]["configured"] is False
        assert client.get("/api/stripe/dashboard-sync").json()["configured"] is False
        assert client.get("/api/email/config").json()["connected"] is False  # the saved mailbox was not reapplied

        # the five one-click demo scenarios, run to the end (HITL approved like a visitor would)
        scenarios = client.get("/api/scenarios").json()
        assert [s["key"] for s in scenarios] == ["duplicate_small", "fraud", "injection", "refund_late", "spanish_refund"]
        outcomes = {}
        for s in scenarios:
            r = client.post("/api/tickets", json={"message": s["message"], "customer_email": s["customer_email"]})
            outcomes[s["key"]] = run_to_end(client, r.json()["ticket_id"])["execution_status"]
        assert outcomes == {
            "duplicate_small": "EXECUTED", "fraud": "EXECUTED", "injection": "EXECUTED",
            "refund_late": "EXECUTED", "spanish_refund": "EXECUTED",
        }

        # inbound email: simulator and provider webhook (a real source, which would normally be answered by SMTP)
        sim = client.post("/api/email/simulate", json={
            "from": "bob@startup.io", "subject": "Double charged", "body_plain": "We were double charged $480 for the annual plan.",
        })
        hook = client.post("/api/webhooks/inbound-email", json={
            "from": "Erin Walsh <erin@designhub.co>", "subject": "Refund request",
            "body_plain": "Hello, please refund my $29 Pro Monthly charge from last week.", "message_id": "<demo-1@mail>",
        })
        for r in (sim, hook):
            assert r.status_code == 202
            run_to_end(client, r.json()["ticket_id"])

        # real mailboxes cannot be connected or tested from the cockpit
        cfg = {"provider": "gmail", "email": "someone@gmail.com", "app_password": "abcd efgh ijkl mnop"}
        assert client.post("/api/email/test-connection", json=cfg).status_code == 403
        assert client.post("/api/email/save-config", json=cfg).status_code == 403
        assert client.post("/api/email/poll-now").status_code == 409

        ledger = client.get("/api/ledger").json()
        assert ledger["refunds"], "the demo still moves (simulated) money in the local ledger"
        assert {m["delivery"] for m in ledger["outbox"]} == {"outbox"}  # recorded, never sent over SMTP
        assert ledger["alerts"] and {a["delivery"] for a in ledger["alerts"]} == {"simulated"}
    assert guard.attempts == [], f"outbound network attempts in demo mode: {guard.attempts}"


def test_demo_rate_limit_is_per_client_ip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env = {"DEMO_MODE": "true", "DEMO_WRITE_LIMIT_PER_MINUTE": "3", "DEMO_CLIENT_IP_HEADERS": "true-client-ip"}
    fault = {"tool": "db.query_transactions", "times": 1, "kind": "timeout"}
    with make_client(monkeypatch, tmp_path, env) as client:
        a, b = {"True-Client-IP": "203.0.113.10"}, {"True-Client-IP": "203.0.113.11"}
        assert [client.post("/api/admin/faults", json=fault, headers=a).status_code for _ in range(3)] == [200, 200, 200]
        blocked = client.post("/api/admin/faults", json=fault, headers=a)
        assert blocked.status_code == 429 and int(blocked.headers["retry-after"]) >= 1
        assert client.post("/api/admin/faults", json=fault, headers=b).status_code == 200  # another visitor
        assert client.get("/api/health", headers=a).status_code == 200  # reads have their own, larger budget
        assert all(client.get("/health", headers=a).status_code == 200 for _ in range(5))  # health check exempt
        big = client.post("/api/email/simulate", content=b"x" * 70_000, headers={**b, "content-type": "application/json"})
        assert big.status_code == 413


def test_client_ip_resolution() -> None:
    scope = {"client": ("10.0.0.5", 1234), "headers": [(b"x-forwarded-for", b"6.6.6.6, 198.51.100.7")]}
    assert client_ip(scope, 1) == "198.51.100.7"  # appended by the trusted proxy; 6.6.6.6 is client-supplied
    assert client_ip(scope, 0) == "10.0.0.5"
    cf = {**scope, "headers": [*scope["headers"], (b"true-client-ip", b"203.0.113.9")]}
    assert client_ip(cf, 1, ("true-client-ip",)) == "203.0.113.9"
    assert client_ip({"headers": []}, 1) == "unknown"


def test_sliding_window_limiter() -> None:
    now = [0.0]
    lim = SlidingWindowLimiter(clock=lambda: now[0], max_keys=2)
    assert [lim.hit("a", write=False, limit=2, write_limit=1) for _ in range(2)] == [None, None]
    assert lim.hit("a", write=False, limit=2, write_limit=1) == pytest.approx(60.0)
    now[0] = 61.0
    assert lim.hit("a", write=True, limit=2, write_limit=1) is None  # window slid
    assert lim.hit("a", write=True, limit=2, write_limit=1) is not None
    lim.hit("b", write=False, limit=2, write_limit=1)
    lim.hit("c", write=False, limit=2, write_limit=1)  # evicts the least recently used key
    assert set(lim._hits) == {"b", "c"}


# ------------------------------------------------------------------ DEMO_MODE=false (default)
HEALTH_KEYS = {
    "status", "version", "llm", "embedder", "hitl_threshold_usd", "judge_pass_threshold", "max_tool_retries",
    "max_redrafts", "email", "autopilot", "stripe",
}


def test_without_demo_mode_the_app_is_unchanged(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with make_client(monkeypatch, tmp_path, {"LLM_PROVIDER": "mock"}) as client:
        assert client.get("/api/demo").json() == {"demo_mode": False}
        assert set(client.get("/api/health").json()) == HEALTH_KEYS
        assert client.get("/api/scenarios").json() == [dict(s) for s in SCENARIOS] and len(SCENARIOS) == 11
        fault = {"tool": "db.query_transactions", "times": 1, "kind": "timeout"}
        assert {client.post("/api/admin/faults", json=fault).status_code for _ in range(30)} == {200}  # no rate limit
        big = client.post("/api/tickets", json={"message": "x" * 70_000})
        assert big.status_code == 422  # the usual validation error, not the demo's 413

        # the mailbox endpoints still reach the (here faked) mail server
        def refuse(host: str, port: int) -> Any:
            raise ConnectionRefusedError(f"{host}:{port}")

        client.app.state.mailbox._imap = refuse  # type: ignore[attr-defined]
        cfg = {"provider": "gmail", "email": "someone@gmail.com", "app_password": "abcd efgh ijkl mnop"}
        r = client.post("/api/email/test-connection", json=cfg)
        assert r.status_code == 200 and r.json()["imap"]["ok"] is False
    assert {s["key"] for s in DEMO_SCENARIOS} <= {s["key"] for s in SCENARIOS} | {"spanish_refund"}
