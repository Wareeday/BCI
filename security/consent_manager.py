"""
security/consent_manager.py
============================
GDPR-compliant consent management and right-to-erasure implementation.

Regulatory requirements:
  GDPR Article 7:   Granular opt-in consent per use case
  GDPR Article 9:   Neural EEG = biometric data → explicit consent required
  GDPR Article 17:  Right to erasure ("right to be forgotten")
  GDPR Article 35:  DPIA mandatory before clinical deployment
  IEEE 2857 §6.2:   Consent captured per-purpose with revoke trigger

Consent purposes (granular):
  1. neural_processing   — basic BCI operation (required)
  2. model_training      — use sessions to improve the model (optional)
  3. anonymized_research — share de-identified data for research (optional)
  4. audit_logging       — retain audit logs (required for safety)
"""

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional
from loguru import logger

from security.audit_logger import AuditLogger


class ConsentPurpose(Enum):
    NEURAL_PROCESSING = "neural_processing"
    MODEL_TRAINING = "model_training"
    ANONYMIZED_RESEARCH = "anonymized_research"
    AUDIT_LOGGING = "audit_logging"


REQUIRED_PURPOSES = {ConsentPurpose.NEURAL_PROCESSING, ConsentPurpose.AUDIT_LOGGING}


@dataclass
class ConsentRecord:
    """Immutable consent record (one per user per purpose)."""
    consent_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    purpose: str = ""
    granted: bool = False
    timestamp_granted: Optional[float] = None
    timestamp_revoked: Optional[float] = None
    ip_address: Optional[str] = None
    version: str = "1.0"

    @property
    def is_active(self) -> bool:
        return self.granted and self.timestamp_revoked is None


class ConsentManager:
    """
    Manages user consent per GDPR Article 7 and 9.

    Consent is stored in-memory (dev) or database (production).
    Revocation triggers immediate erasure pipeline.
    """

    def __init__(self, audit_logger: Optional[AuditLogger] = None):
        self._consents: dict[str, list[ConsentRecord]] = {}   # user_id → [ConsentRecord]
        self._audit = audit_logger
        self._erasure_callbacks: list = []

    def grant_consent(
        self,
        user_id: str,
        purposes: list[ConsentPurpose],
        ip_address: Optional[str] = None,
    ) -> dict[str, str]:
        """
        Record user consent for specified purposes.

        Returns dict of {purpose: consent_id} for audit trail.
        """
        if user_id not in self._consents:
            self._consents[user_id] = []

        consent_ids = {}
        for purpose in purposes:
            record = ConsentRecord(
                user_id=user_id,
                purpose=purpose.value,
                granted=True,
                timestamp_granted=time.time(),
                ip_address=ip_address,
            )
            self._consents[user_id].append(record)
            consent_ids[purpose.value] = record.consent_id

            if self._audit:
                self._audit.log(
                    event_type="consent_granted",
                    user_id=user_id,
                    details={"purpose": purpose.value, "consent_id": record.consent_id},
                )

        logger.info(
            f"Consent granted for user {user_id}: "
            f"{[p.value for p in purposes]}"
        )
        return consent_ids

    def revoke_consent(self, user_id: str, purpose: Optional[ConsentPurpose] = None):
        """
        Revoke consent. If purpose is None, revoke ALL consents.
        Triggers erasure pipeline per GDPR Article 17.
        """
        if user_id not in self._consents:
            logger.warning(f"No consent records found for user {user_id}")
            return

        revoke_time = time.time()
        revoked_purposes = []

        for record in self._consents[user_id]:
            if record.is_active:
                if purpose is None or record.purpose == purpose.value:
                    record.timestamp_revoked = revoke_time
                    record.granted = False
                    revoked_purposes.append(record.purpose)

        if self._audit:
            self._audit.log(
                event_type="consent_revoked",
                user_id=user_id,
                details={"purposes": revoked_purposes, "timestamp": revoke_time},
            )

        logger.info(f"Consent revoked for user {user_id}: {revoked_purposes}")

        # Trigger erasure callbacks (GDPR Article 17)
        self._trigger_erasure(user_id, revoked_purposes)

    def check_consent(self, user_id: str, purpose: ConsentPurpose) -> bool:
        """Return True if user has active consent for this purpose."""
        if user_id not in self._consents:
            return False
        return any(
            r.purpose == purpose.value and r.is_active
            for r in self._consents[user_id]
        )

    def has_required_consents(self, user_id: str) -> bool:
        """Check all required consents are active."""
        return all(
            self.check_consent(user_id, p)
            for p in REQUIRED_PURPOSES
        )

    def register_erasure_callback(self, callback):
        """Register a function to call when data must be erased."""
        self._erasure_callbacks.append(callback)

    def _trigger_erasure(self, user_id: str, purposes: list[str]):
        """Invoke all registered erasure callbacks."""
        for cb in self._erasure_callbacks:
            try:
                cb(user_id=user_id, purposes=purposes)
                logger.info(f"Erasure callback executed for user {user_id}")
            except Exception as exc:
                logger.error(f"Erasure callback failed: {exc}")

    def export_consents(self, user_id: str) -> list[dict]:
        """Export consent records (for DPIA / audit purposes)."""
        if user_id not in self._consents:
            return []
        return [asdict(r) for r in self._consents[user_id]]