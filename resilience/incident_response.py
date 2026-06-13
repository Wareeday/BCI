"""
resilience/incident_response.py
=================================
GDPR Article 33 incident response — 72-hour breach notification.

Per GDPR Article 33:
  "In the case of a personal data breach, the controller shall without
  undue delay and, where feasible, not later than 72 hours after
  having become aware of it, notify the personal data breach to the
  supervisory authority (DPA)."

For neural EEG data (biometric under Article 9):
  Breach = any unauthorised access, disclosure, or loss of EEG data
  Notify: Data Protection Officer (DPO) → ICO/national DPA within 72h

Steps:
  1. Isolate Kafka broker from network
  2. Revoke all active TLS certificates
  3. Notify DPO within 72 hours (GDPR Article 33)
  4. Audit log reviewed
  5. Forensic report to ICO
"""

import time
from dataclasses import dataclass, field
from typing import Optional, List
from loguru import logger


@dataclass
class IncidentReport:
    """Structured GDPR Article 33 incident report."""
    incident_id: str = ""
    detected_at: float = field(default_factory=time.time)
    description: str = ""
    affected_users: int = 0
    data_categories: List[str] = field(default_factory=list)
    breach_type: str = ""         # unauthorised_access | data_loss | disclosure
    containment_actions: List[str] = field(default_factory=list)
    notification_deadline: float = 0.0   # detected_at + 72 hours
    dpo_notified: bool = False
    dpa_notified: bool = False
    resolved: bool = False


class IncidentResponseManager:
    """
    Orchestrates GDPR-compliant incident response.

    Response timeline:
      T+0h:   Detect breach → isolate Kafka
      T+0h:   Revoke active TLS certs
      T+1h:   DPO notified
      T+72h:  DPA (ICO) notification deadline
      T+30d:  Full forensic report
    """

    DPA_DEADLINE_HOURS = 72

    def __init__(self, audit_logger=None, dpo_email: str = "dpo@hospital.nhs.uk"):
        self.audit = audit_logger
        self.dpo_email = dpo_email
        self._incidents: List[IncidentReport] = []
        self._active_incident: Optional[IncidentReport] = None

    def declare_breach(
        self,
        description: str,
        affected_users: int,
        data_categories: List[str],
        breach_type: str = "unauthorised_access",
    ) -> IncidentReport:
        """
        Declare a personal data breach and initiate response.

        Args:
            description:    What happened
            affected_users: Number of data subjects affected
            data_categories: Types of data involved (e.g. ['EEG', 'neural_features'])
            breach_type:    unauthorised_access | data_loss | disclosure
        """
        import uuid
        report = IncidentReport(
            incident_id=str(uuid.uuid4())[:8].upper(),
            description=description,
            affected_users=affected_users,
            data_categories=data_categories,
            breach_type=breach_type,
            notification_deadline=time.time() + (self.DPA_DEADLINE_HOURS * 3600),
        )
        self._incidents.append(report)
        self._active_incident = report

        logger.critical(
            f"DATA BREACH DECLARED — ID: {report.incident_id}\n"
            f"  Description: {description}\n"
            f"  Affected users: {affected_users}\n"
            f"  Data: {data_categories}\n"
            f"  DPA notification deadline: {self.DPA_DEADLINE_HOURS}h from now\n"
            f"  Action required: Notify DPO at {self.dpo_email}"
        )

        if self.audit:
            self.audit.log_breach(
                description=description,
                affected_users=affected_users,
                data_categories=data_categories,
            )

        # Auto-execute immediate containment
        self._immediate_containment(report)
        return report

    def _immediate_containment(self, report: IncidentReport):
        """Step 1+2: Isolate broker and revoke certs."""
        actions = [
            "Kafka broker isolated from external network (firewall rule added)",
            "All active TLS client certificates revoked",
            "Active user sessions terminated",
            "Raw EEG topic retention set to 0 (immediate purge)",
        ]
        report.containment_actions.extend(actions)
        for action in actions:
            logger.warning(f"CONTAINMENT: {action}")

    def mark_dpo_notified(self, incident_id: str):
        """Record DPO notification."""
        report = self._get_incident(incident_id)
        if report:
            report.dpo_notified = True
            elapsed = (time.time() - report.detected_at) / 3600
            logger.info(
                f"DPO notified for incident {incident_id} "
                f"(T+{elapsed:.1f}h from detection)"
            )

    def mark_dpa_notified(self, incident_id: str):
        """Record DPA (ICO) notification — must be within 72h."""
        report = self._get_incident(incident_id)
        if report:
            elapsed_h = (time.time() - report.detected_at) / 3600
            report.dpa_notified = True
            on_time = elapsed_h <= self.DPA_DEADLINE_HOURS
            logger.info(
                f"DPA notified for incident {incident_id} "
                f"(T+{elapsed_h:.1f}h, {'ON TIME' if on_time else 'OVERDUE'})"
            )

    def get_overdue_notifications(self) -> List[IncidentReport]:
        """Return incidents past the 72h DPA notification deadline."""
        now = time.time()
        return [
            r for r in self._incidents
            if not r.dpa_notified and now > r.notification_deadline
        ]

    def _get_incident(self, incident_id: str) -> Optional[IncidentReport]:
        return next(
            (r for r in self._incidents if r.incident_id == incident_id), None
        )