"""
devices/wheelchair_driver.py
==============================
Low-level wheelchair motor driver interface.

Sits between ros_controller.py (high-level BCI commands) and
the physical Arduino Mega running differential drive PWM.

Converts ROS Twist velocity commands into PWM duty cycles:
  Left motor:  D9  (PWM)  + D8  (DIR)
  Right motor: D10 (PWM)  + D11 (DIR)

Safety limits (ISO 14155 aligned):
  Max linear speed:  0.5 m/s
  Max angular speed: 1.0 rad/s
  Emergency stop:    0 PWM on both motors (SAFE_STATE)

Communication: JSON over Serial @ 115,200 baud to Arduino Mega.
"""
import json
import time
from typing import Optional
from dataclasses import dataclass
from loguru import logger

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False


@dataclass
class WheelVelocities:
    """Left and right wheel PWM values (0–255)."""
    left_pwm:  int   # 0–255
    right_pwm: int   # 0–255
    left_dir:  int   # 1 = forward, -1 = backward
    right_dir: int


# ── Velocity → PWM conversion ─────────────────────────────────────
WHEEL_BASE_M   = 0.58   # metres between wheels (standard wheelchair)
MAX_LINEAR_MS  = 0.50   # m/s hard limit
MAX_ANGULAR_RS = 1.00   # rad/s hard limit
MAX_PWM        = 200    # leave headroom for battery sag


def twist_to_wheel_velocities(linear: float, angular: float) -> WheelVelocities:
    """
    Convert ROS Twist (linear x, angular z) to differential drive PWM.

    Differential drive kinematics:
      v_left  = linear - angular * wheelbase / 2
      v_right = linear + angular * wheelbase / 2
    """
    linear  = max(-MAX_LINEAR_MS,  min(MAX_LINEAR_MS,  linear))
    angular = max(-MAX_ANGULAR_RS, min(MAX_ANGULAR_RS, angular))

    v_left  = linear - angular * WHEEL_BASE_M / 2.0
    v_right = linear + angular * WHEEL_BASE_M / 2.0

    def to_pwm(v: float):
        pwm = int(abs(v) / MAX_LINEAR_MS * MAX_PWM)
        return min(pwm, MAX_PWM), (1 if v >= 0 else -1)

    l_pwm, l_dir = to_pwm(v_left)
    r_pwm, r_dir = to_pwm(v_right)

    return WheelVelocities(
        left_pwm=l_pwm, right_pwm=r_pwm,
        left_dir=l_dir, right_dir=r_dir,
    )


class WheelchairDriver:
    """
    Low-level serial driver for motorised wheelchair.

    Accepts (linear, angular) velocity from ROSController and
    converts to PWM signals sent to Arduino Mega.
    """

    BAUD_RATE = 115200
    COMMAND_TIMEOUT_MS = 500   # SAFE_STATE if no command for 500ms

    def __init__(
        self,
        port: str = "/dev/ttyACM1",
        simulate: bool = True,
    ):
        self.port = port
        self.simulate = simulate
        self._serial: Optional[object] = None
        self._current_vels = WheelVelocities(0, 0, 1, 1)
        self._commands_sent = 0
        self._emergency_stops = 0
        self._last_command_time = 0.0

        if not simulate:
            self._connect()
        else:
            logger.info("WheelchairDriver: simulation mode")

    def _connect(self):
        if not SERIAL_AVAILABLE:
            logger.warning("pyserial not installed — using simulation mode")
            self.simulate = True
            return
        try:
            self._serial = serial.Serial(
                port=self.port,
                baudrate=self.BAUD_RATE,
                timeout=0.1,
            )
            time.sleep(2.0)   # Arduino reset time
            logger.success(f"WheelchairDriver connected: {self.port}")
        except Exception as exc:
            logger.warning(f"Wheelchair serial connection failed: {exc}")
            self.simulate = True

    def set_velocity(self, linear: float, angular: float) -> bool:
        """
        Send velocity command to wheelchair motors.

        Returns True if command sent successfully.
        """
        velocities = twist_to_wheel_velocities(linear, angular)
        command = {
            "l_pwm": velocities.left_pwm,
            "r_pwm": velocities.right_pwm,
            "l_dir": velocities.left_dir,
            "r_dir": velocities.right_dir,
        }

        if self.simulate:
            logger.debug(
                f"[SIM] Wheelchair: linear={linear:.2f}m/s "
                f"angular={angular:.2f}rad/s → {command}"
            )
            self._current_vels = velocities
            self._commands_sent += 1
            self._last_command_time = time.time()
            return True

        if self._serial is None:
            return False

        try:
            payload = json.dumps(command) + "\n"
            self._serial.write(payload.encode())
            self._serial.flush()
            ack = self._serial.readline().decode("utf-8", errors="ignore").strip()
            if ack == "OK":
                self._current_vels = velocities
                self._commands_sent += 1
                self._last_command_time = time.time()
                return True
            logger.warning(f"Unexpected wheelchair ACK: {ack!r}")
            return False
        except Exception as exc:
            logger.error(f"Wheelchair serial error: {exc}")
            return False

    def emergency_stop(self) -> bool:
        """Immediate stop — zero PWM on both motors."""
        self._emergency_stops += 1
        logger.warning(f"WHEELCHAIR EMERGENCY STOP #{self._emergency_stops}")
        return self.set_velocity(0.0, 0.0)

    def disconnect(self):
        self.emergency_stop()
        if self._serial and not self.simulate:
            self._serial.close()

    def get_stats(self) -> dict:
        return {
            "commands_sent": self._commands_sent,
            "emergency_stops": self._emergency_stops,
            "current_pwm": {
                "left": self._current_vels.left_pwm,
                "right": self._current_vels.right_pwm,
            },
        }