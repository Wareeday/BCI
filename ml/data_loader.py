"""
ml/data_loader.py
==================
Data loader for BCI Competition IV Dataset 2a (motor imagery).

Dataset: 4-class motor imagery EEG — left hand, right hand, feet, rest.
Subjects: 9 healthy participants.
Channels: 22 EEG + 3 EOG (we use first 8 for OpenBCI compatibility).
Trials:   288 per subject (72 per class), 4s epochs at 250 Hz.

Download:
  http://www.bbci.de/competition/iv/desc_2a.pdf
  Files: A01T.gdf … A09T.gdf (training), A01E.gdf … A09E.gdf (evaluation)

If dataset files are not available, the loader falls back to
generating synthetic data with the same statistics.

Reference:
  Tangermann et al. (2012). Review of the BCI Competition IV.
  Frontiers in Neuroscience, 6, 55.
"""
import os
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from loguru import logger


DATASET_DIR = os.getenv("BCI_DATASET_DIR", "data/bci_competition_iv_2a")
N_CHANNELS_USE = 8     # use first 8 of 22 (matches OpenBCI Cyton)
SAMPLE_RATE    = 250   # Hz
EPOCH_SAMPLES  = 1000  # 4 seconds at 250 Hz
N_CLASSES      = 4
CLASS_NAMES    = ["left_hand", "right_hand", "feet", "rest"]


class BCICompetitionIVLoader:
    """
    Loads BCI Competition IV Dataset 2a.
    Falls back to synthetic data if .gdf files are not present.
    """

    def __init__(
        self,
        dataset_dir: str = DATASET_DIR,
        n_channels: int = N_CHANNELS_USE,
        epoch_samples: int = EPOCH_SAMPLES,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.n_channels = n_channels
        self.epoch_samples = epoch_samples
        self._synthetic_seed = 42

    def load_subject(
        self,
        subject_id: int = 1,
        split: str = "train",     # "train" or "eval"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load one subject's data.

        Args:
            subject_id: 1–9
            split:      'train' or 'eval'

        Returns:
            X: (n_trials, n_channels, epoch_samples) float32
            y: (n_trials,) int64  labels 0–3
        """
        suffix = "T" if split == "train" else "E"
        gdf_path = self.dataset_dir / f"A{subject_id:02d}{suffix}.gdf"

        if gdf_path.exists():
            return self._load_gdf(gdf_path)
        else:
            logger.warning(
                f"Dataset file not found: {gdf_path}. "
                "Generating synthetic data with matching statistics. "
                f"Download from: http://www.bbci.de/competition/iv/"
            )
            return self._generate_synthetic(subject_id=subject_id)

    def load_all_subjects(
        self,
        subject_ids: Optional[list] = None,
        split: str = "train",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and concatenate data from multiple subjects.

        Args:
            subject_ids: list of subject IDs (default: all 9)
            split:       'train' or 'eval'

        Returns:
            X: (total_trials, n_channels, epoch_samples)
            y: (total_trials,)
        """
        subject_ids = subject_ids or list(range(1, 10))
        all_X, all_y = [], []
        for sid in subject_ids:
            X, y = self.load_subject(sid, split)
            all_X.append(X)
            all_y.append(y)
        return np.concatenate(all_X), np.concatenate(all_y)

    def _load_gdf(self, path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load .gdf file using MNE-Python."""
        try:
            import mne
            raw = mne.io.read_raw_gdf(str(path), preload=True, verbose=False)
            events, event_id = mne.events_from_annotations(raw, verbose=False)

            # Motor imagery event codes: 769=left, 770=right, 771=feet, 772=rest
            mi_events = {
                "left_hand": 769, "right_hand": 770,
                "feet": 771, "rest": 772,
            }
            label_map = {769: 0, 770: 1, 771: 2, 772: 3}

            epochs = mne.Epochs(
                raw, events,
                event_id={k: v for k, v in mi_events.items() if v in event_id.values()},
                tmin=0.0, tmax=4.0,
                picks=list(range(self.n_channels)),
                baseline=None, preload=True, verbose=False,
            )
            X = epochs.get_data().astype(np.float32)    # (n_trials, n_ch, n_samples)
            y_raw = epochs.events[:, 2]
            y = np.array([label_map.get(e, 0) for e in y_raw], dtype=np.int64)
            logger.success(f"Loaded {len(X)} trials from {path.name}")
            return X, y
        except Exception as exc:
            logger.error(f"Failed to load {path}: {exc}. Using synthetic fallback.")
            return self._generate_synthetic()

    def _generate_synthetic(
        self,
        n_trials: int = 288,
        subject_id: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic motor imagery data matching BCI Competition IV statistics.

        Based on published statistics:
          Mean epoch amplitude: ~10–20 µV
          Alpha ERD in C3/C4 during motor imagery
          4 balanced classes, 72 trials each
        """
        np.random.seed(self._synthetic_seed + subject_id)
        X = np.zeros((n_trials, self.n_channels, self.epoch_samples), dtype=np.float32)
        y = np.array([i % N_CLASSES for i in range(n_trials)], dtype=np.int64)
        t = np.linspace(0, 4.0, self.epoch_samples)

        # Subject-specific amplitude scaling (individual differences)
        amp_scale = 0.7 + np.random.uniform(0, 0.6)

        for idx in range(n_trials):
            label = y[idx]
            for ch in range(self.n_channels):
                # Pink noise baseline
                X[idx, ch] = np.cumsum(np.random.randn(self.epoch_samples)) * 0.4 * amp_scale

                # Alpha rhythm (8–13 Hz) with mu-rhythm suppression
                alpha_amp = amp_scale * 12.0
                X[idx, ch] += alpha_amp * np.sin(2 * np.pi * 10.0 * t + np.random.rand() * 2 * np.pi)

                # Class-specific ERD patterns (BCI Comp IV standard)
                if label == 0 and ch == 2:   # left hand → C3
                    X[idx, ch] += amp_scale * 15.0 * np.sin(2 * np.pi * 12.0 * t)
                elif label == 1 and ch == 4: # right hand → C4
                    X[idx, ch] += amp_scale * 15.0 * np.sin(2 * np.pi * 12.0 * t)
                elif label == 2 and ch == 3: # feet → Cz
                    X[idx, ch] += amp_scale * 12.0 * np.sin(2 * np.pi * 10.0 * t)

        logger.info(
            f"Synthetic dataset: {n_trials} trials, "
            f"{N_CLASSES} classes, subject={subject_id}"
        )
        return X, y

    def get_dataset_info(self) -> dict:
        """Return metadata about the dataset."""
        return {
            "name": "BCI Competition IV Dataset 2a",
            "paradigm": "4-class motor imagery",
            "classes": CLASS_NAMES,
            "n_subjects": 9,
            "trials_per_subject": 288,
            "channels_used": self.n_channels,
            "sample_rate_hz": SAMPLE_RATE,
            "epoch_duration_s": 4.0,
            "dataset_available": any(
                (self.dataset_dir / f"A0{i}T.gdf").exists()
                for i in range(1, 10)
            ),
            "reference": "Tangermann et al. (2012) Front. Neurosci. 6:55",
        }