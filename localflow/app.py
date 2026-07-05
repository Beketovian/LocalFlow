"""The LocalFlow controller - ties recording, transcription, formatting,
context, dictionary, history and injection together.

The controller is UI-free and platform-free: hotkeys, tray icons and real
microphones live outside and call into it. That keeps the entire dictation
flow unit-testable.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from .audio import ArrayRecorder, Recorder, normalize_rms
from .commands import CommandProcessor
from .config import Config
from .context import ActiveWindowProvider, WindowInfo, match_profile
from .dictionary import PersonalDictionary
from .engines.base import STTEngine
from .formatting import format_transcript, smart_join
from .history import History
from .injector import CallbackInjector, Injector
from .sounds import SoundPlayer
from .vad import SilenceDetector, trim_silence


@dataclass
class DictationEvent:
    """Emitted after every completed dictation."""

    raw_text: str
    formatted_text: str
    injected: bool
    app: str = ""
    language: str = ""
    duration: float = 0.0
    mode: str = "dictation"
    elapsed: float = 0.0


@dataclass
class ControllerState:
    status: str = "idle"  # idle | recording | transcribing
    mode: str = "dictation"
    hands_free: bool = False
    last_event: Optional[DictationEvent] = None
    session_text: str = ""


class FlowController:
    def __init__(
        self,
        config: Optional[Config] = None,
        engine: Optional[STTEngine] = None,
        recorder: Optional[Recorder] = None,
        injector: Optional[Injector] = None,
        history: Optional[History] = None,
        window_provider: Optional[ActiveWindowProvider] = None,
        sounds: Optional[SoundPlayer] = None,
        command_processor: Optional[CommandProcessor] = None,
    ) -> None:
        self.config = config or Config()
        self._engine = engine
        self.recorder = recorder or ArrayRecorder()
        self.injector = injector or CallbackInjector()
        self.window_provider = window_provider or ActiveWindowProvider()
        self.sounds = sounds or SoundPlayer(enabled=self.config.audio.feedback_sounds)
        self.commands = command_processor or CommandProcessor()
        self.dictionary = PersonalDictionary(
            words=self.config.dictionary, replacements=self.config.replacements
        )
        if history is not None:
            self.history = history
        elif self.config.save_history:
            self.history = History(self.config.resolved_data_dir() / "history.db")
        else:
            self.history = History(":memory:")

        self.state = ControllerState()
        self.listeners: List[Callable[[DictationEvent], None]] = []
        self.partial_listeners: List[Callable[[str], None]] = []
        self.status_listeners: List[Callable[[str], None]] = []
        self._lock = threading.RLock()
        self._record_started_at = 0.0
        self._active_window = WindowInfo()

    # ------------------------------------------------------------- lifecycle

    @property
    def engine(self) -> STTEngine:
        if self._engine is None:
            from .engines.registry import create_engine

            self._engine = create_engine(self.config)
        return self._engine

    def on_dictation(self, listener: Callable[[DictationEvent], None]) -> None:
        self.listeners.append(listener)

    def on_partial(self, listener: Callable[[str], None]) -> None:
        self.partial_listeners.append(listener)

    def on_status(self, listener: Callable[[str], None]) -> None:
        """Subscribe to status changes: idle | recording | transcribing."""
        self.status_listeners.append(listener)

    def _set_status(self, status: str) -> None:
        self.state.status = status
        for listener in self.status_listeners:
            try:
                listener(status)
            except Exception:
                pass

    # ------------------------------------------------------------ dictation

    def start_recording(self, mode: str = "dictation") -> bool:
        with self._lock:
            if self.state.status != "idle":
                return False
            self._set_status("recording")
            self.state.mode = mode
        # capture the focused window *before* the user switches attention
        self._active_window = self.window_provider.get()
        self._record_started_at = time.time()
        self.recorder.start()
        self.sounds.start()
        return True

    def stop_recording(self) -> Optional[DictationEvent]:
        with self._lock:
            if self.state.status != "recording":
                return None
            self._set_status("transcribing")
        self.sounds.stop()
        audio = self.recorder.stop()
        try:
            event = self._process_audio(audio, mode=self.state.mode)
        finally:
            with self._lock:
                self._set_status("idle")
        return event

    def cancel_recording(self) -> None:
        with self._lock:
            if self.state.status != "recording":
                return
            self._set_status("idle")
        self.recorder.stop()

    def dictate_array(self, audio: np.ndarray, mode: str = "dictation") -> DictationEvent:
        """One-shot dictation from an in-memory buffer (file mode, tests)."""
        self._active_window = self.window_provider.get()
        self._record_started_at = time.time()
        return self._process_audio(audio, mode=mode)

    # ----------------------------------------------------------- processing

    def _process_audio(self, audio: np.ndarray, mode: str) -> DictationEvent:
        t0 = time.time()
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        duration = audio.shape[0] / self.config.audio.sample_rate

        audio = trim_silence(audio, threshold=self.config.audio.vad_threshold / 2)
        if self.config.audio.normalize:
            audio = normalize_rms(audio, target_rms=self.config.audio.target_rms)

        language = self.config.engine.language
        result = self.engine.transcribe(
            audio,
            language=None if language == "auto" else language,
            initial_prompt=self.dictionary.initial_prompt(),
        )
        raw = result.clean_text

        window = self._active_window
        profile = match_profile(window, self.config.profiles)
        overrides = dict(profile.overrides) if profile else {}

        if mode == "command":
            formatted = raw  # command instructions are used verbatim
        else:
            corrected = self.dictionary.correct(raw)
            formatted = format_transcript(corrected, self.config.formatting, overrides)

        injected = False
        if formatted and mode != "command":
            text_out = smart_join(self.state.session_text, formatted)
            if self.config.output.trailing_space and not text_out.endswith(("\n", " ")):
                out = text_out + " "
            else:
                out = text_out
            try:
                self.injector.inject(out)
                injected = True
                self.state.session_text += out
            except Exception:
                self.sounds.error()

        event = DictationEvent(
            raw_text=raw,
            formatted_text=formatted,
            injected=injected,
            app=window.app or window.title,
            language=result.language or "",
            duration=duration,
            mode=mode,
            elapsed=time.time() - t0,
        )
        self.state.last_event = event

        if formatted:
            self.history.add(
                raw_text=raw,
                formatted_text=formatted,
                app=event.app,
                language=event.language,
                duration=duration,
                mode=mode,
            )
        for listener in self.listeners:
            try:
                listener(event)
            except Exception:
                pass
        return event

    def reset_session(self) -> None:
        """Forget session text (smart-join context)."""
        self.state.session_text = ""

    # ----------------------------------------------------- hands-free mode

    def run_hands_free_once(
        self,
        poll_interval: float = 0.1,
        max_seconds: Optional[float] = None,
    ) -> Optional[DictationEvent]:
        """Record until trailing silence, then process. Blocking."""
        detector = SilenceDetector(
            threshold=self.config.audio.vad_threshold,
            stop_after_seconds=self.config.audio.silence_stop_after,
            sample_rate=self.config.audio.sample_rate,
        )
        if not self.start_recording(mode="hands-free"):
            return None
        limit = max_seconds or self.config.audio.max_recording_seconds
        start = time.time()
        seen = 0
        try:
            while time.time() - start < limit:
                time.sleep(poll_interval)
                snapshot = self.recorder.snapshot()
                fresh = snapshot[seen:]
                seen = snapshot.shape[0]
                if detector.feed(fresh):
                    break
        except KeyboardInterrupt:
            pass
        return self.stop_recording()

    # -------------------------------------------------------- command mode

    def run_command(self, instruction: str, selected_text: str) -> Optional[str]:
        """Apply a spoken instruction to selected text; inject the result."""
        edited = self.commands.apply(instruction, selected_text)
        if edited is None:
            self.sounds.error()
            return None
        try:
            self.injector.inject(edited)
        except Exception:
            self.sounds.error()
            return None
        self.history.add(
            raw_text=instruction,
            formatted_text=edited,
            app=self._active_window.app,
            mode="command",
        )
        return edited

    def close(self) -> None:
        self.history.close()
        if self._engine is not None:
            self._engine.close()
