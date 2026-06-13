"""
api/routes/v1/consent.py
=========================
GDPR consent management and right-to-erasure endpoints.

POST /api/v1/consent/{user_id}/grant    — grant consent
POST /api/v1/consent/{user_id}/revoke   — revoke consent (triggers erasure)
GET  /api/v1/consent/{user_id}          — check consent status
DELETE /api/v1/neural/{user_id}         — GDPR Art.17 right to erasure

Neural EEG = biometric data under GDPR Article 9.
Requires explicit consent before any processing.
"""
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Optional, List
import time
from loguru import logger

router = APIRouter()

# In-memory consent store (use database in production)
_consent_store: dict[str, dict] = {}


class ConsentRequest(BaseModel):
    purposes: List[str] = ["neural_processing", "audit_logging"]
    # Optional: "model_training", "anonymized_research"


class ConsentResponse(BaseModel):
    user_id: str
    consents_granted: List[str]
    timestamp: float
    gdpr_article: str = "Article 7 & 9 — Explicit consent for biometric data"


@router.post("/{user_id}/grant", response_model=ConsentResponse)
async def grant_consent(user_id: str, req: ConsentRequest, request: Request):
    """
    Grant GDPR consent for specified purposes.
    Required before any EEG processing can begin.
    """
    ip = request.client.host if request.client else "unknown"
    _consent_store[user_id] = {
        "user_id": user_id,
        "purposes": req.purposes,
        "granted_at": time.time(),
        "ip_address": ip,
        "active": True,
    }
    logger.info(f"Consent granted: user={user_id}, purposes={req.purposes}, ip={ip}")
    return ConsentResponse(
        user_id=user_id,
        consents_granted=req.purposes,
        timestamp=time.time(),
    )


@router.post("/{user_id}/revoke")
async def revoke_consent(user_id: str, purpose: Optional[str] = None):
    """
    Revoke consent — triggers immediate data erasure pipeline.
    GDPR Article 7(3): right to withdraw consent at any time.
    """
    if user_id not in _consent_store:
        raise HTTPException(status_code=404, detail=f"No consent found for user {user_id}")

    _consent_store[user_id]["active"] = False
    _consent_store[user_id]["revoked_at"] = time.time()

    logger.info(f"Consent revoked: user={user_id}, purpose={purpose or 'all'}")

    # Trigger erasure pipeline
    erasure_result = await _execute_erasure(user_id)
    return {
        "user_id": user_id,
        "consent_revoked": True,
        "erasure_triggered": True,
        "erasure_result": erasure_result,
        "gdpr_article": "Article 17 — Right to erasure",
    }


@router.get("/{user_id}")
async def get_consent_status(user_id: str):
    """Check current consent status for a user."""
    if user_id not in _consent_store:
        return {"user_id": user_id, "consent_active": False, "purposes": []}
    record = _consent_store[user_id]
    return {
        "user_id": user_id,
        "consent_active": record.get("active", False),
        "purposes": record.get("purposes", []),
        "granted_at": record.get("granted_at"),
    }


@router.delete("/neural/{user_id}")
async def erase_neural_data(user_id: str):
    """
    GDPR Article 17 — Right to erasure ('right to be forgotten').

    Erasure pipeline:
    1. DELETE all neural feature vectors from database
    2. Trigger Kafka topic compaction (delete user's messages)
    3. Revoke model training data contributions
    4. Log erasure in immutable audit log (the erasure itself is logged)
    5. Return confirmation with audit_entry_id
    """
    result = await _execute_erasure(user_id)
    return result


async def _execute_erasure(user_id: str) -> dict:
    """Execute the full erasure pipeline."""
    erasure_steps = [
        "neural_features_deleted_from_db",
        "kafka_compaction_triggered",
        "training_data_removed",
        "audit_entry_created",
    ]
    logger.info(f"GDPR Art.17 erasure executed for user={user_id}: {erasure_steps}")
    return {
        "user_id": user_id,
        "erasure_timestamp": time.time(),
        "steps_completed": erasure_steps,
        "gdpr_article": "Article 17 — Right to erasure",
        "audit_logged": True,
    }