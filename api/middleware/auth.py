"""
api/middleware/auth.py
======================
JWT-based authentication middleware.

Public endpoints (no auth required):
  GET /health
  GET /
  GET /docs
  GET /redoc
  GET /openapi.json

All other endpoints require:
  Authorization: Bearer <JWT_TOKEN>

Token generation: POST /api/v1/auth/token (username + password)

In production: integrate with hospital IAM (OAuth2 + OIDC).
"""

import os
from typing import Optional
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from jose import JWTError, jwt
from loguru import logger

SECRET_KEY = os.getenv("API_SECRET_KEY", "dev-secret-change-in-production")
ALGORITHM = os.getenv("API_ALGORITHM", "HS256")

PUBLIC_PATHS = {"/", "/health", "/docs", "/redoc", "/openapi.json", "/ws/eeg",
                "/ws/commands", "/ws/status", "/api/v1/consent"}


class AuthMiddleware(BaseHTTPMiddleware):
    """Validate JWT Bearer token on protected endpoints."""

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Allow public paths and WebSocket upgrades
        if any(path.startswith(p) for p in PUBLIC_PATHS):
            return await call_next(request)

        # Extract Bearer token
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            # Dev mode: allow without auth when no secret set
            if SECRET_KEY == "dev-secret-change-in-production":
                return await call_next(request)
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing Authorization header"},
            )

        token = auth_header.split(" ", 1)[1]
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            request.state.user_id = payload.get("sub", "anonymous")
        except JWTError as exc:
            logger.warning(f"JWT validation failed: {exc}")
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or expired token"},
            )

        return await call_next(request)


def create_access_token(user_id: str, expires_minutes: int = 60) -> str:
    """Create a signed JWT access token."""
    from datetime import datetime, timedelta
    payload = {
        "sub": user_id,
        "exp": datetime.utcnow() + timedelta(minutes=expires_minutes),
        "iat": datetime.utcnow(),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)