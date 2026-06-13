"""
neurofeedback/adaptive_difficulty.py
======================================
Adaptive difficulty controller for neurofeedback training.

From presentation Step 5 (Adapt):
  "Difficulty auto-adjusts: if >80% sessions successful,
   threshold tightened by 10%"

Design:
  - If accuracy > 80%: increase threshold by adjust_step (harder)
  - If accuracy < 50%: decrease threshold by adjust_step (easier)
  - Threshold clamped to [MIN_THRESHOLD, MAX_THRESHOLD]
"""


class AdaptiveDifficultyController:
    """
    Adjusts neurofeedback threshold based on user performance.
    """

    MIN_THRESHOLD = 0.2
    MAX_THRESHOLD = 0.9

    def __init__(
        self,
        initial_threshold: float = 0.5,
        success_rate_target: float = 0.80,
        adjust_step: float = 0.10,
        lower_bound: float = 0.50,
    ):
        self.current_threshold = initial_threshold
        self.success_rate_target = success_rate_target
        self.adjust_step = adjust_step
        self.lower_bound = lower_bound
        self._adjustments: list[dict] = []

    def adjust(self, session_accuracy: float) -> float:
        """
        Adjust threshold based on last N trials accuracy.

        Returns new threshold.
        """
        old = self.current_threshold
        direction = "unchanged"

        if session_accuracy >= self.success_rate_target:
            # Too easy — tighten by 10%
            self.current_threshold = min(
                self.current_threshold + self.adjust_step,
                self.MAX_THRESHOLD,
            )
            direction = "increased (harder)"
        elif session_accuracy < self.lower_bound:
            # Too hard — loosen
            self.current_threshold = max(
                self.current_threshold - self.adjust_step,
                self.MIN_THRESHOLD,
            )
            direction = "decreased (easier)"

        self._adjustments.append({
            "accuracy": session_accuracy,
            "old_threshold": old,
            "new_threshold": self.current_threshold,
            "direction": direction,
        })
        return self.current_threshold

    def get_history(self) -> list[dict]:
        return self._adjustments