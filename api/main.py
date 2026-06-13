"""
api/main.py
===========
FastAPI REST + WebSocket application entry point.

Endpoints:
  POST /api/v1/sessions/start     — start BCI session
  POST /api/v1/sessions/stop      — stop session
  GET  /api/v1/status             — system health
  GET  /api/v1/eeg/latest         — latest EEG sample
  POST /api/v1/commands/override  — manual override
  POST /api/v1/consent/{user_id}  — grant/revoke consent
  DELETE /api/v1/neural/{user_id} — GDPR Article 17 erasure
  GET  /api/v1/audit              — audit log query
  WS   /ws/eeg                    — real-time EEG WebSocket
  WS   /ws/commands               — real-time command WebSocket
"""

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

# Load environment
load_dotenv()

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.routes.v1 import eeg, devices, ml, consent, sessions, audit
from api.websocket_handler import router as ws_router
from api.middleware.auth import AuthMiddleware
from security.audit_logger import AuditLogger
from database.session import init_db


# ── Shared app state ──────────────────────────────────────────────
class AppState:
    audit_logger: AuditLogger = None
    bci_session_active: bool = False
    total_requests: int = 0


app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    # ── Startup ───────────────────────────────────────────────────
    logger.info("BCI Platform API starting up...")

    # Initialise audit logger
    app_state.audit_logger = AuditLogger(
        log_file=os.getenv("AUDIT_LOG_FILE", "logs/audit.log"),
    )
    app_state.audit_logger.log(
        event_type="system_startup",
        details={"version": "1.0.0", "env": os.getenv("ENV", "development")},
    )

    # Initialise database
    await init_db()
    logger.success("Database initialised")

    logger.success("BCI Platform API ready — http://localhost:8000/docs")
    yield

    # ── Shutdown ──────────────────────────────────────────────────
    logger.info("BCI Platform API shutting down...")
    app_state.audit_logger.log(event_type="system_shutdown")


# ── FastAPI application ───────────────────────────────────────────
app = FastAPI(
    title="BCI Platform API",
    description=(
        "Brain-Computer Interface Platform — OpenBCI + GNU Radio + Kafka + CNN.\n\n"
        "Standards: IEEE 2857 · ISO 14155 · FDA 510(k) · GDPR"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── CORS ─────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8050", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Auth middleware ───────────────────────────────────────────────
app.add_middleware(AuthMiddleware)

# ── Routers ──────────────────────────────────────────────────────
app.include_router(sessions.router, prefix="/api/v1/sessions", tags=["Sessions"])
app.include_router(eeg.router,      prefix="/api/v1/eeg",      tags=["EEG"])
app.include_router(devices.router,  prefix="/api/v1/devices",  tags=["Devices"])
app.include_router(ml.router,       prefix="/api/v1/ml",        tags=["ML"])
app.include_router(consent.router,  prefix="/api/v1/consent",  tags=["Consent/GDPR"])
app.include_router(audit.router,    prefix="/api/v1/audit",    tags=["Audit"])
app.include_router(ws_router,       tags=["WebSocket"])


@app.get("/health", tags=["System"])
async def health():
    """System health check — used by Docker healthcheck and monitoring."""
    return {
        "status": "healthy",
        "session_active": app_state.bci_session_active,
        "audit_entries": app_state.audit_logger.entry_count if app_state.audit_logger else 0,
    }


@app.get("/", tags=["System"])
async def root():
    return {
        "name": "BCI Platform API",
        "version": "1.0.0",
        "docs": "/docs",
        "standards": ["IEEE 2857", "ISO 14155", "FDA 510(k)", "GDPR"],
    }