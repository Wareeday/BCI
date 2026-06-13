"""
neurofeedback/session_tracker.py
==================================
Tracks BCI training progress across multiple neurofeedback sessions.

Persists session history to JSON file so progress is maintained
between application restarts.

Session accuracy progression (from presentation):
  Session 1:  62%
  Session 3:  71%
  Session 5:  79%
  Session 10: 88%
  Session 20: 93%

Used by:
  - dashboard/neurofeedback_panel.py (progress chart)
  - ml/adaptive_calibration.py (accuracy alerts)
  - api/routes/v1/sessions.py (session management)
"""

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
from loguru import logger


@dataclass
class SessionRecord:
    """One completed neurofeedback/BCI session."""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8].upper())
    user_id: str = ""
    paradigm: str = "motor_imagery"       # motor_imagery | p300 | neurofeedback
    started_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None
    duration_min: float = 0.0
    total_trials: int = 0
    accuracy: float = 0.0
    score: int = 0
    calibration_time_min: float = 0.0
    adaptive_retrains: int = 0
    badges_earned: list = field(default_factory=list)
    notes: str = ""

    def close(self):
        self.ended_at = time.time()
        self.duration_min = (self.ended_at - self.started_at) / 60.0


class SessionTracker:
    """
    Persists session history and computes progress statistics.

    Storage: logs/sessions_{user_id}.json (JSON Lines format)
    One record per line, append-only.
    """

    SESSIONS_DIR = Path("logs/sessions")

    def __init__(self, user_id: str):
        self.user_id = user_id
        self._history: list[SessionRecord] = []
        self._current: Optional[SessionRecord] = None
        self.SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        self._file = self.SESSIONS_DIR / f"{user_id}.jsonl"
        self._load()

    # ── Session lifecycle ─────────────────────────────────────────

    def start_session(
        self,
        paradigm: str = "motor_imagery",
    ) -> SessionRecord:
        """Start a new session and return its record."""
        self._current = SessionRecord(
            user_id=self.user_id,
            paradigm=paradigm,
        )
        logger.info(
            f"Session started: {self._current.session_id} "
            f"(user={self.user_id}, paradigm={paradigm})"
        )
        return self._current

    def end_session(
        self,
        accuracy: float = 0.0,
        total_trials: int = 0,
        score: int = 0,
        badges: Optional[list] = None,
        adaptive_retrains: int = 0,
        calibration_time_min: float = 0.0,
    ) -> SessionRecord:
        """Finalise current session and persist to file."""
        if self._current is None:
            raise RuntimeError("No active session. Call start_session() first.")

        self._current.close()
        self._current.accuracy = accuracy
        self._current.total_trials = total_trials
        self._current.score = score
        self._current.badges_earned = badges or []
        self._current.adaptive_retrains = adaptive_retrains
        self._current.calibration_time_min = calibration_time_min

        self._history.append(self._current)
        self._append_to_file(self._current)

        session_num = len(self._history)
        logger.success(
            f"Session {session_num} ended: "
            f"accuracy={accuracy:.1%}, trials={total_trials}, "
            f"duration={self._current.duration_min:.1f}min"
        )

        completed = self._current
        self._current = None
        return completed

    # ── Statistics & progress ─────────────────────────────────────

    def get_accuracy_progression(self) -> list[float]:
        """Return list of session accuracies in chronological order."""
        return [s.accuracy for s in self._history]

    def get_session_count(self) -> int:
        return len(self._history)

    def get_latest_accuracy(self) -> Optional[float]:
        if not self._history:
            return None
        return self._history[-1].accuracy

    def get_best_accuracy(self) -> Optional[float]:
        if not self._history:
            return None
        return max(s.accuracy for s in self._history)

    def get_total_training_time_min(self) -> float:
        return sum(s.duration_min for s in self._history)

    def get_summary(self) -> dict:
        """Return progress summary for dashboard display."""
        if not self._history:
            return {"message": "No sessions recorded yet."}
        return {
            "user_id": self.user_id,
            "total_sessions": len(self._history),
            "latest_accuracy": self.get_latest_accuracy(),
            "best_accuracy": self.get_best_accuracy(),
            "total_training_min": round(self.get_total_training_time_min(), 1),
            "accuracy_progression": self.get_accuracy_progression(),
            "target_met": (self.get_latest_accuracy() or 0.0) >= 0.80,
            "progression_note": self._progression_note(),
        }

    def _progression_note(self) -> str:
        """Generate a human-readable progress note."""
        acc = self.get_latest_accuracy() or 0.0
        n = len(self._history)
        if acc >= 0.93:
            return f"Excellent — Session {n}: {acc:.0%}. Master level reached! 🏆"
        elif acc >= 0.80:
            return f"Good progress — Session {n}: {acc:.0%}. Keep training."
        else:
            return f"Early stage — Session {n}: {acc:.0%}. Improvement expected with practice."

    # ── Persistence ───────────────────────────────────────────────

    def _append_to_file(self, record: SessionRecord):
        """Append session record as a JSON line (append-only)."""
        try:
            with open(self._file, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(record)) + "\n")
        except OSError as exc:
            logger.error(f"Failed to write session record: {exc}")

    def _load(self):
        """Load historical sessions from file."""
        if not self._file.exists():
            return
        try:
            with open(self._file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        self._history.append(SessionRecord(**data))
            logger.debug(
                f"Loaded {len(self._history)} sessions for user {self.user_id}"
            )
        except Exception as exc:
            logger.warning(f"Could not load session history: {exc}")