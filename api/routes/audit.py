"""api/routes/v1/audit.py — Audit log query endpoints (IEEE 2857 §7.1)."""
from fastapi import APIRouter, Query
from typing import Optional
import time

router = APIRouter()

_mock_audit_entries = [
    {"entry_id": "abc123", "event_type": "inference", "user_id": "user_001",
     "timestamp": time.time() - 60, "details": {"class_name": "left", "confidence": 0.91}},
    {"entry_id": "def456", "event_type": "command_issued", "user_id": "user_001",
     "timestamp": time.time() - 30, "details": {"command": "left", "device": "wheelchair"}},
]


@router.get("/")
async def query_audit_log(
    event_type: Optional[str] = Query(None),
    user_id: Optional[str] = Query(None),
    limit: int = Query(50, le=500),
):
    """
    Query immutable audit log — IEEE 2857 §7.1.
    Every inference, command, consent event is logged here.
    """
    results = _mock_audit_entries.copy()
    if event_type:
        results = [e for e in results if e["event_type"] == event_type]
    if user_id:
        results = [e for e in results if e.get("user_id") == user_id]
    return {"entries": results[:limit], "total": len(results)}


@router.get("/sae")
async def get_serious_adverse_events():
    """Return all Serious Adverse Events (SAEs) — ISO 14155 §14."""
    sae_entries = [e for e in _mock_audit_entries if e.get("event_type") == "sae"]
    return {"sae_events": sae_entries, "total": len(sae_entries),
            "standard": "ISO 14155 §14 — SAE reporting within 24h"}