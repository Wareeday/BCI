"""
acquisition/simulator.py
==========================
Standalone EEG simulator for offline development and testing.

Generates realistic EEG with:
  - Pink (1/f) noise baseline
  - Alpha rhythm 8–13 Hz (occipital)
  - Beta rhythm 13–30 Hz (frontal/motor)
  - P300 ERP every 2 seconds (parietal Cz/Pz)
  - Motor imagery ERD on C3/C4
  - 50 Hz powerline noise
  - Occasional eye-blink artefacts

Usage:
  sim = EEGSimulator(paradigm="motor_imagery")
  for sample in sim.stream(duration_s=10):
      process(sample)
"""
import time
import numpy as np
from typing import Generator
from acquisition.openbci_board import EEGSample

CHANNEL_LABELS = ["Fp1", "Fp2", "C3", "Cz", "C4", "P3", "P4", "Oz"]
SAMPLE_RATE = 250


class EEGSimulator:
    """
    High-fidelity EEG simulator for BCI development without hardware.

    Used by:
      - scripts/demo_simulation.py
      - tests/conftest.py (mock_board fixture)
      - scripts/run_pipeline.py --simulate
    """

    def __init__(
        self,
        sample_rate: int = 250,
        n_channels: int = 8,
        paradigm: str = "motor_imagery",
        seed: int = 42,
    ):
        self.sample_rate = sample_rate
        self.n_channels = n_channels
        self.paradigm = paradigm
        np.random.seed(seed)
        self._t = 0.0
        self._sample_id = 0
        self._next_p300 = 2.0
        self._next_blink = np.random.uniform(5.0, 15.0)
        self._mi_class = 0   # rotates every 4 seconds

    def next_sample(self) -> EEGSample:
        """Generate one EEG sample at the current time point."""
        dt = 1.0 / self.sample_rate
        channels = np.zeros(self.n_channels, dtype=np.float32)

        # 1/f pink noise
        for harmonic in [1, 2, 4, 8, 16]:
            channels += (8.0 / harmonic) * np.random.randn(self.n_channels).astype(np.float32)

        # Alpha (8–13 Hz) — occipital channels 6, 7
        alpha = 15.0 * np.sin(2 * np.pi * 10.0 * self._t)
        channels[6] += alpha
        channels[7] += alpha * 0.8

        # Beta (13–30 Hz) — motor cortex C3, Cz, C4
        if self.paradigm == "motor_imagery":
            mi_class = int(self._t / 4.0) % 4
            if mi_class == 0:   # left hand → C3 ERD
                channels[2] += 10.0 * np.sin(2 * np.pi * 20.0 * self._t)
            elif mi_class == 1:  # right hand → C4 ERD
                channels[4] += 10.0 * np.sin(2 * np.pi * 20.0 * self._t)
            elif mi_class == 2:  # feet → Cz
                channels[3] += 9.0 * np.sin(2 * np.pi * 18.0 * self._t)

        # P300 ERP (parietal, Cz=ch3, Pz=ch5)
        if self._t >= self._next_p300:
            delay = self._t - self._next_p300
            if 0.0 <= delay <= 0.6:
                p300 = 20.0 * np.exp(-((delay - 0.3) ** 2) / (2 * 0.05 ** 2))
                channels[3] += p300
                channels[5] += p300 * 0.8
            if delay > 0.6:
                self._next_p300 = self._t + 2.0

        # 50 Hz powerline noise (small)
        channels += 3.0 * np.sin(2 * np.pi * 50.0 * self._t)

        # Eye blink artefact (frontal Fp1, Fp2)
        if self._t >= self._next_blink:
            blink_delay = self._t - self._next_blink
            if blink_delay < 0.15:
                blink = 150.0 * np.exp(-((blink_delay - 0.05) ** 2) / (2 * 0.02 ** 2))
                channels[0] += blink
                channels[1] += blink * 0.9
            else:
                self._next_blink = self._t + np.random.uniform(5.0, 15.0)

        sample = EEGSample(
            timestamp=time.time(),
            channels=channels,
            sample_id=self._sample_id % 256,
        )
        self._t += dt
        self._sample_id += 1
        return sample

    def stream(self, duration_s: float = 10.0) -> Generator[EEGSample, None, None]:
        """Yield samples in real time for the given duration."""
        dt = 1.0 / self.sample_rate
        end_time = time.time() + duration_s
        while time.time() < end_time:
            yield self.next_sample()
            time.sleep(dt)

    def generate_epoch(
        self, n_samples: int = 200, label: int = 0
    ) -> np.ndarray:
        """Generate one complete epoch (n_channels × n_samples)."""
        self._mi_class = label
        epoch = np.zeros((self.n_channels, n_samples), dtype=np.float32)
        for i in range(n_samples):
            s = self.next_sample()
            epoch[:, i] = s.channels
        return epoch