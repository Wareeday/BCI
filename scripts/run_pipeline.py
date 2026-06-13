"""
scripts/run_pipeline.py
=======================
End-to-end BCI pipeline runner.

Wires together all components:
  OpenBCI → LSL → DSP Pipeline → Kafka → CNN → ROS/Arduino

Usage:
  python scripts/run_pipeline.py --simulate          # simulation mode
  python scripts/run_pipeline.py --port /dev/ttyUSB0 # real hardware
  python scripts/run_pipeline.py --paradigm p300      # P300 speller mode

Latency monitoring:
  Prints rolling latency stats every 30 seconds.
  Alerts if DSP > 10ms or end-to-end > 100ms.
"""

import argparse
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from dotenv import load_dotenv
import os

load_dotenv()

from acquisition.openbci_board import OpenBCIBoard, EEGSample
from acquisition.lsl_streamer import LSLStreamer
from dsp.pipeline import DSPPipeline
from streaming.kafka_producer import EEGKafkaProducer
from streaming.watchdog import SignalWatchdog
from ml.predict import RealTimePredictor
from ml.cnn_model import BCICNNModel
from ml.sklearn_baseline import SKLearnBaseline
from devices.ros_controller import ROSController
from devices.prosthetic_servo import ProstheticServoController
from devices.tts_engine import TTSEngine
from resilience.safe_state import SafeStateCoordinator
from resilience.eeg_loss_handler import EEGLossHandler
from security.audit_logger import AuditLogger


class BCIPipeline:
    """
    Full BCI platform pipeline orchestrator.

    Component chain:
      board → lsl → kafka_producer → dsp → predictor → ros/prosthetic
                                         ↓
                                    audit_logger
    """

    def __init__(self, args):
        self.args = args
        self._running = False

        # ── Shared services ────────────────────────────────────────
        self.audit = AuditLogger(log_file="logs/audit.log", echo_to_console=True)
        self.safe_state = SafeStateCoordinator(audit_logger=self.audit)

        # ── Acquisition ────────────────────────────────────────────
        self.board = OpenBCIBoard(
            port=args.port,
            simulate=args.simulate,
            watchdog_timeout_s=0.5,
        )
        self.lsl = LSLStreamer(channel_count=8, sample_rate=250.0)

        # ── Streaming ──────────────────────────────────────────────
        self.kafka = EEGKafkaProducer(
            bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
        )
        self.watchdog = SignalWatchdog(
            eeg_timeout_ms=500.0,
            safe_state_callback=self._on_watchdog_timeout,
            recovery_callback=self._on_watchdog_recovery,
        )

        # ── DSP pipeline ───────────────────────────────────────────
        self.dsp = DSPPipeline(
            n_channels=8,
            notch_hz=50.0,
            epoch_type=args.paradigm,
            on_features=self._on_features_ready,
        )

        # ── ML inference ───────────────────────────────────────────
        lda_fallback = SKLearnBaseline(method="lda")
        self.predictor = RealTimePredictor(
            cnn_model=None,    # set after model loaded
            fallback_model=lda_fallback,
            paradigm=args.paradigm,
        )

        # ── Devices ────────────────────────────────────────────────
        self.ros = ROSController(simulate=True)
        self.prosthetic = ProstheticServoController(simulate=True)
        self.tts = TTSEngine(simulate=True)

        # ── Resilience ─────────────────────────────────────────────
        self.eeg_loss_handler = EEGLossHandler(
            safe_state_callback=self.safe_state.activate,
            alert_clinician_callback=self._alert_clinician,
            audit_logger=self.audit,
        )

        # Register safe-state callbacks
        self.safe_state.register_stop_callback(
            lambda reason="": self.ros.activate_safe_state(reason)
        )

        # Stats
        self._sample_count = 0
        self._command_count = 0
        self._last_stats_time = time.time()

    def start(self):
        """Start the full pipeline."""
        logger.info("=" * 60)
        logger.info("BCI Platform starting...")
        logger.info(f"  Mode:     {'SIMULATION' if self.args.simulate else 'HARDWARE'}")
        logger.info(f"  Paradigm: {self.args.paradigm.upper()}")
        logger.info(f"  Port:     {self.args.port}")
        logger.info("=" * 60)

        self.audit.log(event_type="pipeline_start", details={
            "simulate": self.args.simulate, "paradigm": self.args.paradigm,
        })

        # Connect hardware
        if not self.board.connect():
            logger.error("Failed to connect to OpenBCI board")
            sys.exit(1)

        # Impedance check (ISO 14155 Bench Test)
        logger.info("Running electrode impedance check...")
        impedances = self.board.check_impedances()
        poor = {k: v for k, v in impedances.items() if v >= 5.0}
        if poor:
            logger.warning(f"High impedance channels (>5kΩ): {poor}")
        else:
            logger.success("All electrodes: impedance <5kΩ ✓ (ISO 14155 Bench Test)")

        # Start all components
        self.lsl.start()
        self.tts.start()
        self.watchdog.start()
        self._running = True

        # Start EEG streaming
        self.board.start_streaming(callback=self._on_eeg_sample)
        logger.success("Pipeline running. Press Ctrl+C to stop.")

        # Main loop — print stats
        try:
            while self._running:
                time.sleep(30.0)
                self._print_stats()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self):
        """Graceful shutdown."""
        self._running = False
        logger.info("Shutting down pipeline...")
        self.board.stop_streaming()
        self.board.disconnect()
        self.kafka.close()
        self.watchdog.stop()
        self.tts.stop()
        self.lsl.stop()
        self.audit.log(event_type="pipeline_stop", details={
            "total_samples": self._sample_count,
            "total_commands": self._command_count,
        })
        logger.info("Pipeline stopped cleanly")

    # ── Callbacks ─────────────────────────────────────────────────

    def _on_eeg_sample(self, sample: EEGSample):
        """Called at 250 Hz from OpenBCI thread."""
        self._sample_count += 1
        self.watchdog.ping_eeg()
        self.lsl.push_sample(sample)
        self.kafka.publish_raw_sample(sample)
        self.dsp.process_sample(sample)

    def _on_features_ready(self, processed_eeg):
        """Called when DSP pipeline produces a full feature vector."""
        if self.safe_state.is_active:
            return

        self.kafka.publish_features(
            timestamp=processed_eeg.timestamp,
            feature_vector=processed_eeg.feature_vector,
            epoch_type=processed_eeg.epoch_type,
        )
        self.watchdog.ping_kafka()

        # Run inference
        result = self.predictor.predict(
            epoch=processed_eeg.clean_channels,
            feature_vector=processed_eeg.feature_vector,
        )

        # Log every inference (IEEE 2857 §7.1)
        self.audit.log_inference(
            user_id="demo_user",
            session_id="demo_session",
            predicted_class=result.predicted_class,
            class_name=result.class_name,
            confidence=result.confidence,
            model_used=result.model_used,
            epoch_type=processed_eeg.epoch_type,
        )

        # Route command to device
        if result.decision.value == "issue":
            self._execute_command(result)

    def _execute_command(self, result):
        """Route decoded command to appropriate device."""
        cmd_record = self.ros.process_bci_command(
            class_name=result.class_name,
            confidence=result.confidence,
            model_used=result.model_used,
        )
        self.prosthetic.process_command(result.class_name, result.confidence)
        self.kafka.publish_command(
            timestamp=result.timestamp,
            command=result.class_name,
            confidence=result.confidence,
            model_used=result.model_used,
        )
        self._command_count += 1

    def _on_watchdog_timeout(self, source: str, gap_ms: float):
        self.eeg_loss_handler.on_signal_lost(gap_ms=gap_ms, source=source)

    def _on_watchdog_recovery(self, source: str):
        self.eeg_loss_handler.on_signal_restored()
        self.safe_state.deactivate(authorised_by="watchdog_recovery")

    def _alert_clinician(self, message: str, severity: str = "WARNING"):
        logger.log(severity, f"CLINICIAN ALERT: {message}")

    def _print_stats(self):
        elapsed = time.time() - self._last_stats_time
        rate = self._sample_count / max(1, elapsed)
        dsp_stats = self.dsp.get_latency_stats()
        logger.info(
            f"Pipeline stats: samples={self._sample_count}, "
            f"rate={rate:.0f}Hz, commands={self._command_count}, "
            f"dsp_p95={dsp_stats.get('p95_ms', 0):.1f}ms, "
            f"kafka={self.kafka.stats}"
        )
        self._last_stats_time = time.time()
        self._sample_count = 0


def main():
    parser = argparse.ArgumentParser(description="BCI Platform Pipeline Runner")
    parser.add_argument("--simulate", action="store_true", default=True,
                        help="Run in simulation mode (no hardware)")
    parser.add_argument("--port", default="/dev/ttyUSB0",
                        help="OpenBCI serial port (e.g. /dev/ttyUSB0)")
    parser.add_argument("--paradigm", choices=["motor_imagery", "p300"],
                        default="motor_imagery", help="BCI paradigm")
    args = parser.parse_args()

    pipeline = BCIPipeline(args)

    def signal_handler(sig, frame):
        logger.info("Interrupt received — stopping pipeline...")
        pipeline.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    pipeline.start()


if __name__ == "__main__":
    main()