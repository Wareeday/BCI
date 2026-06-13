"""
api/middleware/rate_limiter.py
================================
Request rate limiter middleware.

Prevents abuse of the BCI API — especially the /api/v1/ml/predict
endpoint which is computationally expensive (CNN inference).

Limits (per IP address):
  General endpoints:   100 requests / minute
  ML predict:          60 requests / minute  (1 Hz max)
  Consent/erasure:     10 requests / minute  (GDPR actions)
  WebSocket:           no limit (streaming)

Uses a sliding window token bucket algorithm.
In production: replace with Redis-backed rate limiter for
  distributed deployments (100+ concurrent users).
"""

import time
from collections import defaultdict, deque
from typing import Dict
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from loguru import logger


# ── Rate limit rules (requests per window_seconds) ────────────────
RATE_LIMITS: Dict[str, dict] = {
    "/api/v1/ml/predict":  {"limit": 60,  "window": 60},    # 1 Hz
    "/api/v1/consent":     {"limit": 10,  "window": 60},    # GDPR actions
    "/api/v1/audit":       {"limit": 30,  "window": 60},
    "default":             {"limit": 100, "window": 60},
}

# Endpoints exempt from rate limiting
EXEMPT_PATHS = {"/health", "/", "/docs", "/redoc", "/openapi.json"}


class SlidingWindowRateLimiter:
    """In-memory sliding window rate limiter per IP."""

    def __init__(self):
        # {ip: deque of request timestamps}
        self._windows: Dict[str, deque] = defaultdict(deque)

    def is_allowed(self, ip: str, limit: int, window_s: float) -> bool:
        """Return True if request is within rate limit."""
        now = time.time()
        cutoff = now - window_s
        q = self._windows[ip]

        # Remove expired timestamps
        while q and q[0] < cutoff:
            q.popleft()

        if len(q) >= limit:
            return False

        q.append(now)
        return True

    def remaining(self, ip: str, limit: int, window_s: float) -> int:
        """Return remaining requests allowed in current window."""
        now = time.time()
        cutoff = now - window_s
        q = self._windows[ip]
        active = sum(1 for ts in q if ts >= cutoff)
        return max(0, limit - active)


_limiter = SlidingWindowRateLimiter()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Apply sliding-window rate limits per IP address."""

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Skip exempt paths and WebSocket upgrades
        if (path in EXEMPT_PATHS
                or path.startswith("/ws/")
                or request.headers.get("upgrade", "").lower() == "websocket"):
            return await call_next(request)

        # Get client IP
        ip = "unknown"
        if request.client:
            ip = request.client.host
        # Support X-Forwarded-For (reverse proxy)
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            ip = forwarded.split(",")[0].strip()

        # Determine applicable rate limit
        rule = RATE_LIMITS.get("default")
        for pattern, r in RATE_LIMITS.items():
            if pattern != "default" and path.startswith(pattern):
                rule = r
                break

        limit  = rule["limit"]
        window = rule["window"]

        if not _limiter.is_allowed(ip, limit, window):
            remaining_s = window  # approximate reset time
            logger.warning(
                f"Rate limit exceeded: ip={ip}, path={path}, "
                f"limit={limit}/{window}s"
            )
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Rate limit exceeded",
                    "limit": limit,
                    "window_seconds": window,
                    "retry_after_seconds": remaining_s,
                },
                headers={
                    "Retry-After": str(remaining_s),
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0",
                },
            )

        response = await call_next(request)
        # Add rate limit headers to all responses
        rem = _limiter.remaining(ip, limit, window)
        response.headers["X-RateLimit-Limit"]     = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(rem)
        return response