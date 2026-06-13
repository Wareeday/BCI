"""
neurofeedback/feedback_engine.py
=================================
Real-time neurofeedback engine.

5-step neurofeedback loop (from presentation slide):
  1. Measure:  Alpha/beta power via FFT on 1-second sliding window
  2. Display:  Plotly dashboard / BrainViz shows band power as animated bars
  3. Target:   Clinician sets target band (e.g. increase beta 12-30Hz)
  4. Reward:   Progress bar fills when target met; audio chime; score++
  5. Adapt:    If >80% sessions successful, threshold tightened by 10%

Adaptive performance data (from presentation):
  Session 1:  62%
  Session 3:  71%
  Session 5:  79%
  Session 10: 88%
  Session 20: 93%
"""

import time
from dataclasses import dataclass, field
from typing import Callable, Optional
import numpy as np
from loguru import logger

from neurofeedback.band_power import BandPowerCalculator
from neurofeedback.gamification import GamificationEngine
from neurofeedback.adaptive_difficulty import AdaptiveDifficultyController


@dataclass
class FeedbackState:
    """Current neurofeedback session state."""
    session_id: str = ""
    user_id: str = ""
    target_band: str = "beta"       # alpha | beta | theta | delta
    target_increase: bool = True    # True = increase, False = decrease
    current_power_uv2: float = 0.0
    target_threshold: float = 0.5   # normalised [0, 1]
    threshold_met: bool = False
    score: int = 0
    trial_count: int = 0
    success_count: int = 0
    session_accuracy: float = 0.0
    timestamp: float = field(default_factory=time.time)


class NeurofeedbackEngine:
    """
    Orchestrates the full neurofeedback loop.

    Connects:
      BandPowerCalculator  → FFT-based power estimation
      GamificationEngine   → score, progress bar, chime
      AdaptiveDifficulty   → auto-adjust threshold

    Session performance tracking feeds into AdaptiveCalibration
    (the same SGD re-training loop).
    """

    WINDOW_SECONDS = 1.0    # 1-second sliding FFT window

    def __init__(
        self,
        sample_rate: float = 250.0,
        n_channels: int = 8,
        on_feedback: Optional[Callable[[FeedbackState], None]] = None,
    ):
        self.sample_rate = sample_rate
        self.n_channels = n_channels
        self.on_feedback = on_feedback

        self.band_power = BandPowerCalculator(
            sample_rate=sample_rate,
            window_seconds=self.WINDOW_SECONDS,
        )
        self.gamification = GamificationEngine()
        self.difficulty = AdaptiveDifficultyController(
            initial_threshold=0.5,
            success_rate_target=0.80,
            adjust_step=0.10,
        )

        self._state = FeedbackState()
        self._session_history: list[FeedbackState] = []

    def start_session(
        self,
        user_id: str,
        session_id: str,
        target_band: str = "beta",
        target_increase: bool = True,
    ):
        """Initialise a new neurofeedback session."""
        self._state = FeedbackState(
            session_id=session_id,
            user_id=user_id,
            target_band=target_band,
            target_increase=target_increase,
            target_threshold=self.difficulty.current_threshold,
        )
        self.gamification.reset_session()
        logger.info(
            f"Neurofeedback session started: user={user_id}, "
            f"target={target_band} ({'increase' if target_increase else 'decrease'}), "
            f"threshold={self.difficulty.current_threshold:.2f}"
        )

    def process_eeg(self, eeg_epoch: np.ndarray) -> FeedbackState:
        """
        Process one EEG epoch and update feedback state.

        Args:
            eeg_epoch: (n_channels, n_samples)

        Returns:
            updated FeedbackState
        """
        # ── Step 1: Measure band power ─────────────────────────────
        powers = self.band_power.compute(eeg_epoch)
        target_power = powers.get(self._state.target_band, 0.0)
        normalised = self.band_power.normalise(target_power, self._state.target_band)

        self._state.current_power_uv2 = target_power
        self._state.trial_count += 1

        # ── Step 3 + 4: Check target and reward ────────────────────
        threshold_met = (
            (normalised >= self._state.target_threshold) if self._state.target_increase
            else (normalised <= (1.0 - self._state.target_threshold))
        )
        self._state.threshold_met = threshold_met

        if threshold_met:
            self._state.success_count += 1
            score_delta = self.gamification.on_success(normalised)
            self._state.score += score_delta

        self._state.session_accuracy = (
            self._state.success_count / max(1, self._state.trial_count)
        )

        # ── Step 5: Adapt difficulty ───────────────────────────────
        if self._state.trial_count % 10 == 0 and self._state.trial_count > 0:
            new_threshold = self.difficulty.adjust(self._state.session_accuracy)
            self._state.target_threshold = new_threshold

        self._state.timestamp = time.time()

        if self.on_feedback:
            self.on_feedback(self._state)

        return self._state

    def end_session(self) -> dict:
        """Finalise session and return summary stats."""
        summary = {
            "session_id": self._state.session_id,
            "user_id": self._state.user_id,
            "target_band": self._state.target_band,
            "total_trials": self._state.trial_count,
            "success_count": self._state.success_count,
            "accuracy": self._state.session_accuracy,
            "final_score": self._state.score,
            "final_threshold": self._state.target_threshold,
        }
        self._session_history.append(self._state)
        logger.info(
            f"Session ended: accuracy={self._state.session_accuracy:.2f}, "
            f"score={self._state.score}"
        )
        return summary

    def get_session_history_accuracies(self) -> list[float]:
        """Return accuracy progression across sessions (for dashboard chart)."""
        return [s.session_accuracy for s in self._session_history]