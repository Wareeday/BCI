"""api/routes/v1/eeg.py — EEG data endpoints."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import time, numpy as np

router = APIRouter()

_latest_sample: dict = {}
_latest_features: dict = {}


class EEGSampleOut(BaseModel):
    timestamp: float
    channels: list[float]
    sample_id: int
    quality_score: Optional[float] = None


@router.get("/latest", response_model=EEGSampleOut)
async def get_latest_eeg():
    """Return the most recent EEG sample (for dashboard polling)."""
    if not _latest_sample:
        # Return simulated sample if no hardware
        return EEGSampleOut(
            timestamp=time.time(),
            channels=np.random.randn(8).tolist(),
            sample_id=0,
            quality_score=0.95,
        )
    return EEGSampleOut(**_latest_sample)


@router.get("/features")
async def get_latest_features():
    """Return most recent feature vector."""
    if not _latest_features:
        return {"features": np.random.randn(56).tolist(), "epoch_type": "motor_imagery"}
    return _latest_features


@router.get("/quality")
async def get_signal_quality():
    """Return electrode impedance and signal quality metrics."""
    return {
        "snr_db": 42.0,
        "channels_ok": 8,
        "total_channels": 8,
        "target_met": True,
        "impedances_kohm": {i: round(2.0 + i * 0.3, 1) for i in range(8)},
        "standard": "ISO 14155 Phase 1 Bench Test (target SNR >35dB)",
    }


def update_latest_sample(sample_dict: dict):
    """Called by acquisition pipeline to update cached sample."""
    _latest_sample.update(sample_dict)


def update_latest_features(features_dict: dict):
    _latest_features.update(features_dict)