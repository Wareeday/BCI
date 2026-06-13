"""
security/audit_logger.py
=========================
Immutable audit logger — every inference, access, and export timestamped.

Regulatory requirements:
  FDA 21 CFR Part 11:  Electronic records must be audit-trailed
  IEEE 2857 §7.1:      CNN decision logged per inference (GradCAM on request)
  ISO 14155 §14:       Adverse events (SAE) must be reported immediately
  GDPR Article 30:     Records of processing activities

Log entries are append-only (WORM — Write Once Read Many).
In production: use Azure Immutable Blob Storage or AWS S3 Object Lock.
"""

import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Optional
from loguru import logger


@dataclass
class AuditEntry:
    """One immutable audit log entry."""
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    event_type: str = ""           # consent_granted | inference | command | sae | erasure | breach
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    details: dict = field(default_factory=dict)
    severity: str = "INFO"         # INFO | WARNING | ERROR | CRITICAL (SAE)
    ip_address: Optional[str] = None


class AuditLogger:
    """
    Thread-safe, append-only audit logger.

    Writes JSON Lines (.jsonl) to an audit log file.
    Each line is one AuditEntry — never overwritten or deleted.

    Event types:
      inference       — every CNN prediction (IEEE 2857 §7.1)
      command_issued  — every actuator command (ISO 14155 §14)
      consent_granted — GDPR Article 7
      consent_revoked — GDPR Article 7
      erasure_request — GDPR Article 17
      data_access     — who accessed neural data
      sae             — serious adverse event (ISO 14155 §14)
      breach          — data breach (GDPR Article 33, 72h reporting)
      safe_state      — watchdog SAFE_STATE activations
    """

    def __init__(
        self,
        log_file: str = "logs/audit.log",
        echo_to_console: bool = False,
    ):
        self.log_file = Path(log_file)
        self.echo = echo_to_console
        self._lock = threading.Lock()
        self._entry_count = 0

        # Create log directory
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"AuditLogger: writing to {self.log_file}")

    def log(
        self,
        event_type: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        details: Optional[dict] = None,
        severity: str = "INFO",
        ip_address: Optional[str] = None,
    ) -> str:
        """
        Write one immutable audit entry.

        Returns entry_id for cross-referencing.
        """
        entry = AuditEntry(
            event_type=event_type,
            user_id=user_id,
            session_id=session_id,
            details=details or {},
            severity=severity,
            ip_address=ip_address,
        )

        entry_dict = asdict(entry)
        line = json.dumps(entry_dict, default=str) + "\n"

        with self._lock:
            try:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(line)
                self._entry_count += 1
            except OSError as exc:
                # Critical: audit log write failure must be flagged
                logger.critical(f"AUDIT LOG WRITE FAILURE: {exc}")

        if self.echo or severity in ("ERROR", "CRITICAL"):
            logger.log(severity, f"AUDIT [{event_type}] user={user_id} {details}")

        return entry.entry_id

    def log_inference(
        self,
        user_id: str,
        session_id: str,
        predicted_class: int,
        class_name: str,
        confidence: float,
        model_used: str,
        epoch_type: str,
        gradcam_available: bool = False,
    ) -> str:
        """Log every CNN inference — IEEE 2857 §7.1 requirement."""
        return self.log(
            event_type="inference",
            user_id=user_id,
            session_id=session_id,
            details={
                "predicted_class": predicted_class,
                "class_name": class_name,
                "confidence": round(confidence, 4),
                "model": model_used,
                "epoch_type": epoch_type,
                "gradcam_available": gradcam_available,
            },
        )

    def log_command(
        self,
        user_id: str,
        session_id: str,
        command: str,
        device: str,
        confidence: float,
        executed: bool,
        rejection_reason: Optional[str] = None,
    ) -> str:
        """Log every actuator command — ISO 14155 §14."""
        return self.log(
            event_type="command_issued" if executed else "command_rejected",
            user_id=user_id,
            session_id=session_id,
            severity="INFO" if executed else "WARNING",
            details={
                "command": command,
                "device": device,
                "confidence": round(confidence, 4),
                "executed": executed,
                "rejection_reason": rejection_reason,
            },
        )

    def log_sae(
        self,
        user_id: str,
        session_id: str,
        description: str,
        device: str,
    ) -> str:
        """
        Log a Serious Adverse Event (SAE) — ISO 14155 §14.
        SAE = any unintended wheelchair movement or injury risk.
        Must be reported to IRB within 24 hours.
        """
        entry_id = self.log(
            event_type="sae",
            user_id=user_id,
            session_id=session_id,
            severity="CRITICAL",
            details={
                "description": description,
                "device": device,
                "reporting_deadline_hours": 24,
                "standard": "ISO 14155 §14",
            },
        )
        logger.critical(
            f"SAE LOGGED — ISO 14155 §14: {description} "
            f"(user={user_id}, device={device}). "
            f"Report to IRB within 24 hours. entry_id={entry_id}"
        )
        return entry_id

    def log_breach(
        self,
        description: str,
        affected_users: int,
        data_categories: list,
    ) -> str:
        """
        Log a data breach — GDPR Article 33 (notify DPA within 72 hours).
        """
        entry_id = self.log(
            event_type="breach",
            severity="CRITICAL",
            details={
                "description": description,
                "affected_users": affected_users,
                "data_categories": data_categories,
                "gdpr_reporting_deadline_hours": 72,
                "notify": "Data Protection Officer (DPO)",
            },
        )
        logger.critical(
            f"DATA BREACH: {description}. "
            f"Notify DPO within 72h per GDPR Article 33. entry_id={entry_id}"
        )
        return entry_id

    def query(
        self,
        event_type: Optional[str] = None,
        user_id: Optional[str] = None,
        since: Optional[float] = None,
        limit: int = 100,
    ) -> list[dict]:
        """Read and filter audit log entries."""
        results = []
        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        if event_type and entry.get("event_type") != event_type:
                            continue
                        if user_id and entry.get("user_id") != user_id:
                            continue
                        if since and entry.get("timestamp", 0) < since:
                            continue
                        results.append(entry)
                        if len(results) >= limit:
                            break
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            pass
        return results

    @property
    def entry_count(self) -> int:
        return self._entry_count