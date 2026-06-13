"""
neurofeedback/gamification.py
==============================
Gamification engine for BCI training engagement.

From the presentation:
  Step 4 (Reward): progress bar fills when target met; audio chime; score++
  Purpose: maintain user engagement over 20+ training sessions

Evidence-based design:
  Gamification increases BCI training adherence (Lotte et al., 2013)
  Session 1 → 62%, Session 20 → 93% accuracy progression
"""

import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class SessionStats:
    score: int = 0
    streak: int = 0
    max_streak: int = 0
    badges_earned: list = None

    def __post_init__(self):
        self.badges_earned = []


class GamificationEngine:
    """
    Tracks score, streaks, and badges during a neurofeedback session.

    Score formula:
      Base: +10 per successful trial
      Streak bonus: +5 × streak_length (up to ×5 = +25)
      Precision bonus: +5 if normalised power > 0.8
    """

    STREAK_BONUS_PER_LEVEL = 5
    MAX_STREAK_BONUS = 25
    PRECISION_THRESHOLD = 0.8
    PRECISION_BONUS = 5

    BADGES = {
        "first_success":    {"threshold": 1,   "label": "First Step",     "icon": "🧠"},
        "10_streak":        {"threshold": 10,  "label": "On Fire",        "icon": "🔥"},
        "50_trials":        {"threshold": 50,  "label": "Persistent",     "icon": "💪"},
        "90_accuracy":      {"threshold": 0.9, "label": "Master",         "icon": "🏆"},
        "100_score":        {"threshold": 100, "label": "Century",        "icon": "💯"},
    }

    def __init__(self):
        self._stats = SessionStats()
        self._trial_count = 0
        self._success_count = 0

    def reset_session(self):
        self._stats = SessionStats()
        self._trial_count = 0
        self._success_count = 0

    def on_success(self, normalised_power: float) -> int:
        """
        Called on a successful trial. Returns score delta.
        """
        self._trial_count += 1
        self._success_count += 1
        self._stats.streak += 1
        self._stats.max_streak = max(self._stats.streak, self._stats.max_streak)

        # Base score
        score_delta = 10

        # Streak bonus (capped)
        streak_bonus = min(
            self._stats.streak * self.STREAK_BONUS_PER_LEVEL,
            self.MAX_STREAK_BONUS,
        )
        score_delta += streak_bonus

        # Precision bonus
        if normalised_power >= self.PRECISION_THRESHOLD:
            score_delta += self.PRECISION_BONUS

        self._stats.score += score_delta
        self._check_badges()
        return score_delta

    def on_failure(self):
        """Called on a failed trial — reset streak."""
        self._trial_count += 1
        self._stats.streak = 0

    def _check_badges(self):
        """Award badges based on milestones."""
        accuracy = self._success_count / max(1, self._trial_count)

        checks = [
            ("first_success",  self._success_count >= 1),
            ("10_streak",      self._stats.max_streak >= 10),
            ("50_trials",      self._trial_count >= 50),
            ("90_accuracy",    accuracy >= 0.9),
            ("100_score",      self._stats.score >= 100),
        ]
        for badge_id, condition in checks:
            if condition and badge_id not in self._stats.badges_earned:
                self._stats.badges_earned.append(badge_id)
                badge = self.BADGES[badge_id]

    @property
    def stats(self) -> SessionStats:
        return self._stats

    def get_ui_state(self) -> dict:
        """Return state for dashboard progress bar / gamification panel."""
        accuracy = self._success_count / max(1, self._trial_count)
        return {
            "score": self._stats.score,
            "streak": self._stats.streak,
            "max_streak": self._stats.max_streak,
            "accuracy": round(accuracy, 3),
            "badges": [self.BADGES[b]["label"] for b in self._stats.badges_earned],
            "trial_count": self._trial_count,
        }