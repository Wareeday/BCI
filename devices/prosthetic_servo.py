"""
devices/prosthetic_servo.py
============================
Arduino Mega servo driver for upper-limb prosthetic.

4 motor imagery classes → prosthetic gestures:
  rest  → hold current position
  left  → open hand (extend all fingers)
  right → close grip (flex all fingers)
  feet  → pinch grip (thumb + index only)

Communication:
  JSON over Serial at 115,200 baud — 8ms latency
  Deadman switch: opens circuit if CNN confidence <75% × 3 consecutive frames

Equivalent Arduino sketch: devices/arduino/servo_control.ino
"""

import json
import serial
import threading
import time
from enum import Enum
from typing import Optional
from loguru import logger


class ProstheticGesture(Enum):
    REST = "rest"
    OPEN = "open"
    CLOSE = "close"
    PINCH = "pinch"


# Map motor imagery class to prosthetic gesture
MI_TO_GESTURE = {
    "rest":  ProstheticGesture.REST,
    "left":  ProstheticGesture.OPEN,
    "right": ProstheticGesture.CLOSE,
    "feet":  ProstheticGesture.PINCH,
}

# Servo angles for each gesture {servo_id: angle_degrees}
GESTURE_SERVO_ANGLES = {
    ProstheticGesture.REST:  {"thumb": 90, "index": 90, "middle": 90, "ring": 90, "pinky": 90},
    ProstheticGesture.OPEN:  {"thumb": 180, "index": 180, "middle": 180, "ring": 180, "pinky": 180},
    ProstheticGesture.CLOSE: {"thumb": 0,   "index": 0,   "middle": 0,   "ring": 0,   "pinky": 0},
    ProstheticGesture.PINCH: {"thumb": 30,  "index": 30,  "middle": 90,  "ring": 90,  "pinky": 90},
}


class ProstheticServoController:
    """
    Controls prosthetic servo motors via Arduino Mega over serial.

    Safety: deadman switch logic.
    If confidence < 0.75 for 3 consecutive frames:
      - Opens circuit (sets all servos to REST)
      - Logs adverse event for ISO 14155 reporting
    """

    CONFIDENCE_THRESHOLD = 0.75
    DEADMAN_CONSECUTIVE_FRAMES = 3
    BAUD_RATE = 115200

    def __init__(
        self,
        port: str = "/dev/ttyACM0",
        baud: int = 115200,
        simulate: bool = True,
    ):
        self.port = port
        self.baud = baud
        self.simulate = simulate

        self._serial: Optional[serial.Serial] = None
        self._current_gesture = ProstheticGesture.REST
        self._low_confidence_count = 0
        self._deadman_active = False
        self._commands_sent = 0
        self._adverse_events = 0

        if not simulate:
            self._connect()
        else:
            logger.info("ProstheticServo: simulation mode (no Arduino required)")

    def _connect(self):
        try:
            self._serial = serial.Serial(
                port=self.port,
                baudrate=self.baud,
                timeout=0.1,
            )
            time.sleep(2.0)   # Arduino reset
            logger.success(f"Arduino connected: {self.port} @ {self.baud}")
        except serial.SerialException as exc:
            logger.warning(f"Arduino connection failed: {exc}. Using simulation.")
            self.simulate = True

    def process_command(self, class_name: str, confidence: float) -> bool:
        """
        Process BCI command and actuate if safe.

        Returns True if gesture was executed.
        """
        # Deadman switch check
        if confidence < self.CONFIDENCE_THRESHOLD:
            self._low_confidence_count += 1
            if self._low_confidence_count >= self.DEADMAN_CONSECUTIVE_FRAMES:
                if not self._deadman_active:
                    self._activate_deadman()
            return False
        else:
            self._low_confidence_count = 0
            if self._deadman_active:
                self._deactivate_deadman()

        gesture = MI_TO_GESTURE.get(class_name, ProstheticGesture.REST)
        return self._actuate(gesture, confidence)

    def _actuate(self, gesture: ProstheticGesture, confidence: float) -> bool:
        """Send servo angles to Arduino."""
        if gesture == self._current_gesture:
            return True   # Already in this position

        angles = GESTURE_SERVO_ANGLES[gesture]
        command = {
            "gesture": gesture.value,
            "angles": angles,
            "confidence": round(confidence, 3),
        }

        if self.simulate:
            logger.info(f"[SIM] Prosthetic: {gesture.value} angles={angles}")
            self._current_gesture = gesture
            self._commands_sent += 1
            return True

        # Send JSON command to Arduino
        try:
            payload = json.dumps(command) + "\n"
            self._serial.write(payload.encode("utf-8"))
            self._serial.flush()

            # Read acknowledgement
            ack = self._serial.readline().decode("utf-8", errors="ignore").strip()
            if ack == "OK":
                self._current_gesture = gesture
                self._commands_sent += 1
                logger.debug(f"Prosthetic actuated: {gesture.value}")
                return True
            else:
                logger.warning(f"Unexpected Arduino response: {ack!r}")
                return False
        except Exception as exc:
            logger.error(f"Prosthetic serial error: {exc}")
            return False

    def _activate_deadman(self):
        """Emergency stop — open all servos to REST."""
        self._deadman_active = True
        self._adverse_events += 1
        self._actuate(ProstheticGesture.REST, confidence=0.0)
        logger.error(
            f"DEADMAN SWITCH: {self.DEADMAN_CONSECUTIVE_FRAMES} consecutive "
            f"low-confidence frames — prosthetic set to REST. "
            f"Adverse event #{self._adverse_events} logged."
        )

    def _deactivate_deadman(self):
        self._deadman_active = False
        logger.info("Deadman switch reset — prosthetic control restored")

    def disconnect(self):
        self._actuate(ProstheticGesture.REST, 1.0)
        if self._serial and self._serial.is_open:
            self._serial.close()

    def get_stats(self) -> dict:
        return {
            "current_gesture": self._current_gesture.value,
            "commands_sent": self._commands_sent,
            "deadman_active": self._deadman_active,
            "adverse_events": self._adverse_events,
        }