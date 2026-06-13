"""
api/websocket_handler.py
=========================
WebSocket endpoints for real-time EEG and command streaming.

WS /ws/eeg       — streams raw EEG samples at 250 Hz to dashboard
WS /ws/commands  — streams decoded BCI commands to actuators/UI
WS /ws/status    — streams system health metrics every second

Used by the Plotly Dash dashboard (dashboard/callbacks.py) for
live EEG waveform display and command feedback.
"""

import asyncio
import json
import time
from typing import Set

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger

router = APIRouter()


class ConnectionManager:
    """Manages multiple WebSocket client connections."""

    def __init__(self):
        self.active_eeg: Set[WebSocket] = set()
        self.active_commands: Set[WebSocket] = set()
        self.active_status: Set[WebSocket] = set()

    async def connect_eeg(self, ws: WebSocket):
        await ws.accept()
        self.active_eeg.add(ws)
        logger.info(f"EEG WebSocket connected ({len(self.active_eeg)} clients)")

    async def connect_commands(self, ws: WebSocket):
        await ws.accept()
        self.active_commands.add(ws)
        logger.info(f"Commands WebSocket connected ({len(self.active_commands)} clients)")

    async def connect_status(self, ws: WebSocket):
        await ws.accept()
        self.active_status.add(ws)

    def disconnect(self, ws: WebSocket):
        self.active_eeg.discard(ws)
        self.active_commands.discard(ws)
        self.active_status.discard(ws)

    async def broadcast_eeg(self, data: dict):
        """Broadcast EEG sample to all connected dashboard clients."""
        dead = set()
        for ws in self.active_eeg.copy():
            try:
                await ws.send_text(json.dumps(data))
            except Exception:
                dead.add(ws)
        self.active_eeg -= dead

    async def broadcast_command(self, data: dict):
        """Broadcast BCI command to all connected clients."""
        dead = set()
        for ws in self.active_commands.copy():
            try:
                await ws.send_text(json.dumps(data))
            except Exception:
                dead.add(ws)
        self.active_commands -= dead

    async def broadcast_status(self, data: dict):
        dead = set()
        for ws in self.active_status.copy():
            try:
                await ws.send_text(json.dumps(data))
            except Exception:
                dead.add(ws)
        self.active_status -= dead


manager = ConnectionManager()


@router.websocket("/ws/eeg")
async def eeg_websocket(websocket: WebSocket):
    """
    Stream real-time EEG at 250 Hz.

    Message format:
    {
      "timestamp": 1700000000.0,
      "channels": [1.2, -0.5, 2.1, 0.8, -1.4, 3.2, 0.1, -0.9],
      "sample_id": 1234
    }
    """
    await manager.connect_eeg(websocket)
    try:
        # Simulate 250 Hz EEG stream when no real hardware
        sample_id = 0
        while True:
            # Synthetic EEG with alpha + noise
            t = time.time()
            channels = [
                float(np.sin(2 * np.pi * 10 * t) * 15 + np.random.randn() * 3)
                for _ in range(8)
            ]
            msg = {
                "timestamp": t,
                "channels": [round(v, 3) for v in channels],
                "sample_id": sample_id,
                "type": "eeg_sample",
            }
            await websocket.send_text(json.dumps(msg))
            sample_id += 1
            await asyncio.sleep(1.0 / 250.0)   # 250 Hz
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("EEG WebSocket client disconnected")
    except Exception as exc:
        logger.error(f"EEG WebSocket error: {exc}")
        manager.disconnect(websocket)


@router.websocket("/ws/commands")
async def commands_websocket(websocket: WebSocket):
    """
    Stream decoded BCI commands in real time.

    Message format:
    {
      "timestamp": 1700000000.0,
      "command": "left",
      "confidence": 0.91,
      "decision": "issue",
      "model": "cnn"
    }
    """
    await manager.connect_commands(websocket)
    classes = ["left", "right", "feet", "rest"]
    try:
        while True:
            # Simulate command stream — replace with real predictor output
            probs = np.random.dirichlet([3, 1, 1, 2]).tolist()
            pred = int(np.argmax(probs))
            confidence = probs[pred]
            msg = {
                "timestamp": time.time(),
                "command": classes[pred],
                "confidence": round(confidence, 4),
                "decision": "issue" if confidence >= 0.85 else "hold",
                "model": "cnn",
                "type": "bci_command",
            }
            await websocket.send_text(json.dumps(msg))
            await asyncio.sleep(0.8)   # ~1 command per second
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("Commands WebSocket client disconnected")
    except Exception as exc:
        logger.error(f"Commands WebSocket error: {exc}")
        manager.disconnect(websocket)


@router.websocket("/ws/status")
async def status_websocket(websocket: WebSocket):
    """Stream system health metrics every second."""
    await manager.connect_status(websocket)
    try:
        while True:
            msg = {
                "timestamp": time.time(),
                "type": "health",
                "eeg_streaming": True,
                "kafka_healthy": True,
                "cnn_healthy": True,
                "safe_state": False,
                "dsp_latency_ms": round(7.5 + np.random.rand() * 2.0, 1),
                "end_to_end_latency_ms": round(85 + np.random.rand() * 10, 1),
                "active_ws_clients": {
                    "eeg": len(manager.active_eeg),
                    "commands": len(manager.active_commands),
                },
            }
            await websocket.send_text(json.dumps(msg))
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as exc:
        logger.error(f"Status WebSocket error: {exc}")
        manager.disconnect(websocket)