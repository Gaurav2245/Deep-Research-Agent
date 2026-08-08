"""Lightweight in-memory rate limiting middleware.

Not distributed — counters live in process memory, so they reset on restart
and don't coordinate across multiple processes/instances. That's fine for a
single-instance deployment; swap for a Redis-backed limiter before scaling
out horizontally to more than one API process.
"""
from __future__ import annotations

import time
from threading import Lock
from typing import FrozenSet, Tuple

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Fixed-window rate limiter, keyed per client IP."""

    def __init__(
        self,
        app,
        requests_per_window: int = 60,
        window_seconds: int = 60,
        exempt_paths: FrozenSet[str] = frozenset({"/health", "/ready"}),
    ):
        super().__init__(app)
        self.requests_per_window = requests_per_window
        self.window_seconds = window_seconds
        self.exempt_paths = exempt_paths
        self._lock = Lock()
        self._buckets: dict[str, Tuple[int, float]] = {}

    async def dispatch(self, request: Request, call_next):
        if self.requests_per_window <= 0 or request.url.path in self.exempt_paths:
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.time()

        with self._lock:
            count, window_start = self._buckets.get(client_ip, (0, now))
            if now - window_start >= self.window_seconds:
                count, window_start = 0, now
            count += 1
            self._buckets[client_ip] = (count, window_start)

        if count > self.requests_per_window:
            retry_after = max(0, int(self.window_seconds - (now - window_start)))
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Too many requests",
                    "detail": f"Rate limit exceeded. Try again in {retry_after}s.",
                },
                headers={"Retry-After": str(retry_after)},
            )

        return await call_next(request)
