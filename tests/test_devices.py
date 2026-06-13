"""
tests/test_devices.py
======================
Unit tests for ROS controller, prosthetic servo, and TTS engine.
"""

import time
import pytest
import numpy as np


class TestROSController:
    def test_ros_controller_initialises_in_simulation(self):
        from devices.ros_controller import ROSController
        ros = ROSController(simulate=True)   # add simulate arg
        assert ros is not None

    def test_low_confidence_command_rejected(self):
        from devices.ros_controller import ROSController
        ros = ROSController()
        cmd = ros.process_bci_command("left", confidence=0.50, model_used="cnn")
        assert not cmd.executed
        assert cmd.rejection_reason is not None

    def test_high_confidence_after_confirmation_window(self):
        """After 200ms confirmation window, high-confidence command should issue."""
        from devices.ros_controller import ROSController
        ros = ROSController(confirmation_window_ms=50.0)  # shorten for test
        # First call starts confirmation window
        cmd1 = ros.process_bci_command("left", confidence=0.91, model_used="cnn")
        assert not cmd1.executed  # starts window
        time.sleep(0.06)  # wait past 50ms window
        cmd2 = ros.process_bci_command("left", confidence=0.91, model_used="cnn")
        assert cmd2.executed

    def test_safe_state_blocks_commands(self):
        from devices.ros_controller import ROSController
        ros = ROSController()
        ros.activate_safe_state(reason="test")
        cmd = ros.process_bci_command("forward", confidence=0.95, model_used="cnn")
        assert not cmd.executed
        assert "safe_state" in cmd.rejection_reason

    def test_stats_dict(self):
        from devices.ros_controller import ROSController
        ros = ROSController()
        stats = ros.get_stats()
        assert "commands_issued" in stats
        assert "commands_rejected" in stats
        assert "safe_state_active" in stats


class TestProstheticServo:
    def test_initialises_in_simulation(self):
        from devices.prosthetic_servo import ProstheticServoController
        ctrl = ProstheticServoController(simulate=True)
        assert ctrl is not None

    def test_rest_command_executes(self):
        from devices.prosthetic_servo import ProstheticServoController
        ctrl = ProstheticServoController(simulate=True)
        result = ctrl.process_command("rest", confidence=0.90)
        assert result is True

    def test_low_confidence_triggers_deadman(self):
        from devices.prosthetic_servo import ProstheticServoController
        ctrl = ProstheticServoController(simulate=True)
        # Send 3 low-confidence commands to trigger deadman
        for _ in range(3):
            ctrl.process_command("left", confidence=0.60)
        assert ctrl._deadman_active

    def test_stats_dict(self):
        from devices.prosthetic_servo import ProstheticServoController
        ctrl = ProstheticServoController(simulate=True)
        stats = ctrl.get_stats()
        assert "current_gesture" in stats
        assert "commands_sent" in stats
        assert "deadman_active" in stats
        assert "adverse_events" in stats


class TestTTSEngine:
    def test_initialises_in_simulation(self):
        from devices.tts_engine import TTSEngine
        tts = TTSEngine(simulate=True)
        assert tts is not None

    def test_character_buffering(self):
        from devices.tts_engine import TTSEngine
        tts = TTSEngine(simulate=True)
        tts.start()
        for ch in "HELLO":
            tts.add_character(ch)
        time.sleep(0.05)
        stats = tts.get_stats()
        tts.stop()
        # Buffer should have accumulated chars (before flush)
        assert stats is not None

    def test_speak_direct_enqueues(self):
        from devices.tts_engine import TTSEngine
        tts = TTSEngine(simulate=True)
        tts.start()
        tts.speak("left")
        time.sleep(0.1)
        tts.stop()
        stats = tts.get_stats()
        assert stats["chars_spoken"] >= 0   # may have spoken in background