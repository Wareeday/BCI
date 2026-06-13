"""
database/crud.py
=================
CRUD operations for all database models.

Follows the repository pattern — all DB access goes through here,
never directly from route handlers.

Immutability rule (FDA 21 CFR Part 11):
  AuditLogEntry rows must NEVER be updated or deleted.
  EEGEpoch rows may only be deleted via the erasure API (GDPR Art.17).
"""

import time
from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from loguru import logger

from database.models import (
    User, BCISession, EEGEpoch, BCICommand,
    ConsentRecord, AuditLogEntry, AdverseEvent, MLModelVersion
)


# ── Users ─────────────────────────────────────────────────────────

async def create_user(db: AsyncSession, pseudonym_id: str, role: str = "patient") -> User:
    user = User(pseudonym_id=pseudonym_id, role=role)
    db.add(user)
    await db.flush()
    return user


async def get_user_by_pseudonym(db: AsyncSession, pseudonym_id: str) -> Optional[User]:
    result = await db.execute(select(User).where(User.pseudonym_id == pseudonym_id))
    return result.scalar_one_or_none()


# ── Sessions ──────────────────────────────────────────────────────

async def create_session(
    db: AsyncSession,
    user_id: str,
    paradigm: str = "motor_imagery",
) -> BCISession:
    session = BCISession(user_id=user_id, paradigm=paradigm)
    db.add(session)
    await db.flush()
    return session


async def close_session(
    db: AsyncSession,
    session_id: str,
    accuracy: Optional[float] = None,
    total_epochs: int = 0,
) -> Optional[BCISession]:
    result = await db.execute(select(BCISession).where(BCISession.id == session_id))
    session = result.scalar_one_or_none()
    if session:
        session.ended_at = time.time()
        session.accuracy = accuracy
        session.total_epochs = total_epochs
    return session


# ── EEG Epochs ────────────────────────────────────────────────────

async def store_epoch(
    db: AsyncSession,
    session_id: str,
    pseudonym_id: str,
    timestamp: float,
    epoch_type: str,
    features_encrypted: Optional[str],
    label: Optional[int] = None,
    dsp_latency_ms: Optional[float] = None,
    quality_score: Optional[float] = None,
) -> EEGEpoch:
    """Store anonymised EEG epoch (no raw waveform — IEEE 2857 §5.1)."""
    epoch = EEGEpoch(
        session_id=session_id,
        pseudonym_id=pseudonym_id,
        timestamp=timestamp,
        epoch_type=epoch_type,
        features_encrypted=features_encrypted,
        label=label,
        dsp_latency_ms=dsp_latency_ms,
        quality_score=quality_score,
    )
    db.add(epoch)
    await db.flush()
    return epoch


async def erase_user_epochs(db: AsyncSession, pseudonym_id: str) -> int:
    """
    GDPR Article 17 — delete all epoch records for a pseudonym.
    Returns count of deleted records.
    """
    result = await db.execute(
        select(EEGEpoch).where(EEGEpoch.pseudonym_id == pseudonym_id)
    )
    epochs = result.scalars().all()
    count = len(epochs)
    for epoch in epochs:
        await db.delete(epoch)
    logger.info(f"GDPR erasure: deleted {count} epochs for pseudonym={pseudonym_id}")
    return count


# ── BCI Commands ──────────────────────────────────────────────────

async def log_command(
    db: AsyncSession,
    session_id: str,
    command: str,
    device: str,
    confidence: float,
    model_used: str,
    executed: bool,
    rejection_reason: Optional[str] = None,
) -> BCICommand:
    cmd = BCICommand(
        session_id=session_id,
        command=command,
        device=device,
        confidence=confidence,
        model_used=model_used,
        executed=executed,
        rejection_reason=rejection_reason,
    )
    db.add(cmd)
    await db.flush()
    return cmd


# ── Audit Log ─────────────────────────────────────────────────────

async def append_audit_entry(
    db: AsyncSession,
    event_type: str,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    severity: str = "INFO",
    details: Optional[dict] = None,
) -> AuditLogEntry:
    """Append-only audit entry — never update or delete."""
    entry = AuditLogEntry(
        event_type=event_type,
        user_id=user_id,
        session_id=session_id,
        severity=severity,
        details=details or {},
    )
    db.add(entry)
    await db.flush()
    return entry


async def get_audit_entries(
    db: AsyncSession,
    event_type: Optional[str] = None,
    since: Optional[float] = None,
    limit: int = 100,
) -> List[AuditLogEntry]:
    query = select(AuditLogEntry).order_by(AuditLogEntry.timestamp.desc())
    if event_type:
        query = query.where(AuditLogEntry.event_type == event_type)
    if since:
        query = query.where(AuditLogEntry.timestamp >= since)
    query = query.limit(limit)
    result = await db.execute(query)
    return result.scalars().all()


# ── Adverse Events ────────────────────────────────────────────────

async def log_adverse_event(
    db: AsyncSession,
    description: str,
    session_id: Optional[str] = None,
    device: Optional[str] = None,
) -> AdverseEvent:
    """Log SAE — ISO 14155 §14. Automatically alerts on creation."""
    event = AdverseEvent(
        session_id=session_id,
        description=description,
        device=device,
    )
    db.add(event)
    await db.flush()
    logger.critical(
        f"SAE logged: {description} (device={device}). "
        f"Report to IRB within 24h — ISO 14155 §14."
    )
    return event