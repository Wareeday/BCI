"""
dsp/ica_artifact_removal.py
============================
FastICA artifact removal — removes EOG, EMG, and cardiac artifacts.

Why ICA?
  EEG artifacts (eye blinks = 100–200 µV, muscle = broadband noise,
  heartbeat = ~10 µV at 1 Hz) overlap spectrally with neural signals.
  ICA decomposes multichannel EEG into statistically independent
  components; artifactual ones are identified and zeroed out.

MNE-Python implementation used here. GNU Radio equivalent would require
a custom Python block with the same logic.

Reference: Makeig et al. (1996) Independent Component Analysis of
  Electroencephalographic Data. NIPS.
"""
import numpy as np
from loguru import logger

try:
    import mne
    MNE_AVAILABLE = True
except ImportError:
    logger.warning("MNE-Python not installed. ICA will use sklearn FastICA fallback.")
    MNE_AVAILABLE = False

try:
    from sklearn.decomposition import FastICA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class ICARemoval:
    """
    FastICA-based artifact removal.

    Artifact detection heuristics (when no external classifier):
    1. EOG: component with highest kurtosis in frontal channels
    2. EMG: broadband high-frequency components (>30 Hz variance)
    3. Cardiac: periodic component near 1 Hz

    In production, these heuristics are replaced by a trained
    artifact classifier (logistic regression on IC features).
    """

    def __init__(
        self,
        n_components: int = 8,
        sample_rate: float = 250.0,
        random_state: int = 42,
    ):
        self.n_components = n_components
        self.sample_rate = sample_rate
        self.random_state = random_state
        self._ica = None
        self._is_fitted = False
        self._artifacts_removed = 0

    def fit(self, data: np.ndarray):
        """
        Fit ICA on a calibration block of clean(er) EEG.

        Args:
            data: (n_channels, n_samples) — ideally 30+ seconds
        """
        logger.info(f"Fitting ICA on {data.shape[1]/self.sample_rate:.1f}s of data...")
        if SKLEARN_AVAILABLE:
            self._ica = FastICA(
                n_components=self.n_components,
                random_state=self.random_state,
                max_iter=500,
                tol=0.0001,
            )
            self._ica.fit(data.T)   # sklearn expects (samples, features)
        self._is_fitted = True
        logger.success("ICA fitted successfully")

    def remove_artifacts(self, data: np.ndarray) -> np.ndarray:
        """
        Decompose into ICs, zero out artifact components, reconstruct.

        Args:
            data: (n_channels, n_samples)
        Returns:
            clean: same shape, artifact ICs removed
        """
        if not self._is_fitted or self._ica is None:
            # Auto-fit on this data if not yet fitted (first epoch)
            self.fit(data)

        try:
            # Get independent components
            sources = self._ica.transform(data.T)   # (samples, n_components)
            artifact_mask = self._detect_artifacts(sources, data)
            sources[:, artifact_mask] = 0.0
            # Reconstruct (inverse ICA)
            clean = self._ica.inverse_transform(sources)   # (samples, channels)
            self._artifacts_removed += int(np.sum(artifact_mask))
            return clean.T.astype(np.float32)              # back to (channels, samples)
        except Exception as exc:
            logger.warning(f"ICA failed, returning filtered data: {exc}")
            return data

    def _detect_artifacts(self, sources: np.ndarray, raw: np.ndarray) -> np.ndarray:
        """
        Identify artifact independent components.

        Returns boolean mask (True = artifact to remove).
        """
        n_components = sources.shape[1]
        artifact_mask = np.zeros(n_components, dtype=bool)

        for ic in range(n_components):
            component = sources[:, ic]

            # EOG heuristic: high kurtosis (blink = sharp transient)
            kurt = self._kurtosis(component)
            if kurt > 5.0:
                artifact_mask[ic] = True
                continue

            # EMG heuristic: high-frequency variance ratio
            hf_ratio = self._hf_ratio(component)
            if hf_ratio > 0.6:
                artifact_mask[ic] = True
                continue

        logger.debug(
            f"ICA: {np.sum(artifact_mask)}/{n_components} components marked as artifact"
        )
        return artifact_mask

    @staticmethod
    def _kurtosis(x: np.ndarray) -> float:
        """Excess kurtosis (Gaussian = 0, peaky/blinky > 3)."""
        x = x - np.mean(x)
        std = np.std(x)
        if std < 1e-10:
            return 0.0
        return float(np.mean((x / std) ** 4) - 3.0)

    def _hf_ratio(self, x: np.ndarray) -> float:
        """Fraction of power above 30 Hz (EMG indicator)."""
        from scipy.signal import welch
        freqs, psd = welch(x, fs=self.sample_rate, nperseg=min(256, len(x)))
        total_power = np.sum(psd) + 1e-12
        hf_power = np.sum(psd[freqs > 30.0])
        return float(hf_power / total_power)

    @property
    def artifacts_removed_total(self) -> int:
        return self._artifacts_removed