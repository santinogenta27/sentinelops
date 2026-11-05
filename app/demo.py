"""Public-demo protections: per-IP rate limiting and a request-size cap (active only with DEMO_MODE=true).

Implemented as a pure ASGI middleware, so streaming responses (the SSE event stream) pass through
untouched. With demo mode off it forwards every request as-is: the app behaves exactly as before.
"""

from __future__ import annotations

import json
import math
import time
from collections import OrderedDict, deque
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

from app.config import Settings

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

WRITE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
EXEMPT_PATHS = frozenset({"/health"})  # the platform health check must never be throttled
WINDOW_S = 60.0


def client_ip(scope: Scope, trusted_proxy_hops: int, trusted_headers: tuple[str, ...] = ()) -> str:
    """Client address for rate limiting.

    1. A header set by a trusted edge proxy (e.g. True-Client-IP / CF-Connecting-IP behind Cloudflare), which
       the edge overwrites, so a client cannot forge it.
    2. Otherwise X-Forwarded-For: each trusted proxy appends the address it received the request from, so with
       N trusted hops the client is the N-th entry from the right. Entries further left are client-supplied
       and spoofable, which is why the leftmost value is never used.
    3. Otherwise the TCP peer.
    """
    headers: dict[str, str] = {
        bytes(name).decode("latin-1").lower(): bytes(value).decode("latin-1") for name, value in scope.get("headers") or []
    }
    for name in trusted_headers:
        value = headers.get(name, "").strip()
        if value:
            return value
    if trusted_proxy_hops > 0 and headers.get("x-forwarded-for"):
        hops = [h.strip() for h in headers["x-forwarded-for"].split(",") if h.strip()]
        if len(hops) >= trusted_proxy_hops:
            return hops[-trusted_proxy_hops]
    client = scope.get("client")
    return str(client[0]) if client else "unknown"


class SlidingWindowLimiter:
    """Per-key sliding one-minute windows for all requests and for writes. Bounded memory."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic, max_keys: int = 10_000) -> None:
        self._clock = clock
        self._max_keys = max_keys
        self._hits: OrderedDict[str, tuple[deque[float], deque[float]]] = OrderedDict()

    def hit(self, key: str, *, write: bool, limit: int, write_limit: int) -> float | None:
        """Record a request. Returns None if allowed, else the seconds to wait before retrying."""
        now = self._clock()
        if key in self._hits:
            self._hits.move_to_end(key)  # most recently used last
        else:
            self._hits[key] = (deque(), deque())
        all_hits, writes = self._hits[key]
        while len(self._hits) > self._max_keys:
            self._hits.popitem(last=False)
        for q in (all_hits, writes):
            while q and now - q[0] >= WINDOW_S:
                q.popleft()
        if len(all_hits) >= limit:
            return WINDOW_S - (now - all_hits[0])
        if write and len(writes) >= write_limit:
            return WINDOW_S - (now - writes[0])
        all_hits.append(now)
        if write:
            writes.append(now)
        return None


class DemoGuardMiddleware:
    def __init__(self, app: ASGIApp, *, settings: Callable[[], Settings], limiter: SlidingWindowLimiter | None = None) -> None:
        self.app = app
        self._settings = settings
        self.limiter = limiter or SlidingWindowLimiter()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        s = self._settings()
        if not s.demo_mode or scope.get("path") in EXEMPT_PATHS:
            await self.app(scope, receive, send)
            return
        for name, value in scope.get("headers") or []:
            if name == b"content-length" and value.isdigit() and int(value) > s.demo_max_body_bytes:
                await _reject(send, 413, "request body too large for the public demo", None)
                return
        method = str(scope.get("method", "GET")).upper()
        wait = self.limiter.hit(
            client_ip(scope, s.demo_trusted_proxy_hops, _header_list(s.demo_client_ip_headers)),
            write=method in WRITE_METHODS,
            limit=s.demo_rate_limit_per_minute,
            write_limit=s.demo_write_limit_per_minute,
        )
        if wait is not None:
            await _reject(send, 429, "rate limit reached for the public demo, try again shortly", wait)
            return
        await self.app(scope, receive, send)


def _header_list(raw: str) -> tuple[str, ...]:
    return tuple(h.strip().lower() for h in raw.split(",") if h.strip())


async def _reject(send: Send, status: int, detail: str, retry_after: float | None) -> None:
    body = json.dumps({"detail": detail}).encode()
    headers = [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]
    if retry_after is not None:
        headers.append((b"retry-after", str(max(1, math.ceil(retry_after))).encode()))
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})
