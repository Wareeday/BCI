"""api/routes/v1/devices.py — Device control and status endpoints."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import time

router = APIRouter()

_device_states = {
    "wheelchair": {"state": "stopped", "last_command": None, "safe_state": False},
    "prosthetic": {"gesture": "rest", "safe_state": False},
}


class ManualOverrideRequest(BaseModel):
    device: str
    command: str
    reason: str = "clinician_override"


@router.get("/status")
async def get_device_status():
    return {"devices": _device_states, "timestamp": time.time()}


@router.post("/override")
async def manual_override(req: ManualOverrideRequest):
    """Manual clinician override — takes priority over BCI commands."""
    if req.device not in _device_states:
        raise HTTPException(status_code=404, detail=f"Device '{req.device}' not found")
    _device_states[req.device]["last_command"] = req.command
    return {
        "device": req.device,
        "command_applied": req.command,
        "override_reason": req.reason,
        "timestamp": time.time(),
    }


@router.post("/safe-state")
async def activate_safe_state(reason: str = "manual"):
    """Activate SAFE_STATE on all devices — emergency stop."""
    for device in _device_states:
        _device_states[device]["safe_state"] = True
    return {"safe_state_active": True, "reason": reason, "devices_stopped": list(_device_states)}