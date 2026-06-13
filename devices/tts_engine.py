"""
devices/tts_engine.py
=====================
Offline Text-to-Speech engine using Festival/pyttsx3 on Raspberry Pi 4.

Why offline?
  Fully offline — no cloud dependency.
  Critical for hospital network isolation (patient data must not
  leave the local subnet per GDPR and hospital security policy).

Latency:
  Avg word output: 320ms from final character
  Characters buffered → synthesised as complete words/phrases

Used in P300 speller paradigm:
  User spells characters → buffered into words → spoken aloud
"""

import threading
import time
from collections import deque
from typing import Optional
from loguru import logger

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    logger.warning("pyttsx3 not installed. TTS disabled. pip install pyttsx3")
    PYTTSX3_AVAILABLE = False


class TTSEngine:
    """
    Offline TTS with character buffering.

    Buffers individual characters from P300 speller output.
    Speaks complete words when space is detected or buffer timeout reached.
    """

    WORD_TIMEOUT_S = 2.0    # speak partial word after 2s of no new chars

    def __init__(self, rate: int = 150, volume: float = 0.9, simulate: bool = False):
        self.rate = rate
        self.volume = volume
        self.simulate = simulate

        self._engine = None
        self._char_buffer: deque = deque()
        self._word_buffer = ""
        self._last_char_time = 0.0
        self._speech_thread: Optional[threading.Thread] = None
        self._speech_queue: deque = deque()
        self._stop_event = threading.Event()
        self._chars_spoken = 0

        if not simulate and PYTTSX3_AVAILABLE:
            self._init_engine()

    def _init_engine(self):
        try:
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", self.rate)
            self._engine.setProperty("volume", self.volume)
            voices = self._engine.getProperty("voices")
            if voices:
                self._engine.setProperty("voice", voices[0].id)
            logger.success(f"TTS engine initialised (rate={self.rate}, volume={self.volume})")
        except Exception as exc:
            logger.warning(f"TTS engine init failed: {exc}. Using simulation.")
            self.simulate = True

    def start(self):
        """Start background TTS processing thread."""
        self._stop_event.clear()
        self._speech_thread = threading.Thread(
            target=self._tts_loop,
            daemon=True,
            name="TTSEngine",
        )
        self._speech_thread.start()
        logger.info("TTS engine started")

    def stop(self):
        self._stop_event.set()
        if self._speech_thread:
            self._speech_thread.join(timeout=3.0)

    def add_character(self, char: str):
        """
        Add one character from P300 speller.
        Called every time the P300 classifier identifies a character.
        """
        self._char_buffer.append(char)
        self._last_char_time = time.time()

        # Flush on space or punctuation
        if char in (" ", ".", "?", "!"):
            self._flush_word()

    def speak(self, text: str):
        """Directly enqueue text for speech (bypasses char buffer)."""
        self._speech_queue.append(text)
        logger.debug(f"TTS enqueued: {text!r}")

    def _flush_word(self):
        """Speak accumulated word buffer."""
        word = "".join(self._char_buffer).strip()
        if word:
            self._speech_queue.append(word)
            self._chars_spoken += len(word)
        self._char_buffer.clear()

    def _tts_loop(self):
        """Background thread: speak queued text."""
        while not self._stop_event.is_set():
            # Auto-flush on timeout
            if (self._char_buffer and
                    time.time() - self._last_char_time > self.WORD_TIMEOUT_S):
                self._flush_word()

            if self._speech_queue:
                text = self._speech_queue.popleft()
                self._say(text)

            time.sleep(0.05)

    def _say(self, text: str):
        """Synthesise and speak one phrase."""
        t_start = time.time()
        if self.simulate or not PYTTSX3_AVAILABLE:
            logger.info(f"[TTS SIM] '{text}'")
            time.sleep(len(text) * 0.05)    # simulate speech duration
            return
        try:
            self._engine.say(text)
            self._engine.runAndWait()
            elapsed = (time.time() - t_start) * 1000.0
            logger.debug(f"TTS spoke '{text}' in {elapsed:.0f}ms")
        except Exception as exc:
            logger.error(f"TTS speak error: {exc}")

    def get_stats(self) -> dict:
        return {
            "chars_spoken": self._chars_spoken,
            "queue_depth": len(self._speech_queue),
            "buffer": "".join(self._char_buffer),
        }