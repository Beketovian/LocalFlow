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
from .formatting import detect_send_command, format_transcript, smart_join
from .history import History
from .injector import CallbackInjector, Injector
from .llm import LLMClient
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
    llm_used: bool = False
    stt_seconds: float = 0.0
    llm_seconds: float = 0.0
    auto_sent: bool = False  # voice action "send it" pressed Enter


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
        llm: Optional[LLMClient] = None,
    ) -> None:
        self.config = config or Config()
        self._engine = engine
        self.recorder = recorder or ArrayRecorder()
        self.injector = injector or CallbackInjector()
        self.window_provider = window_provider or ActiveWindowProvider()
        self.sounds = sounds or SoundPlayer(
            enabled=self.config.audio.feedback_sounds,
            sounds_dir=self.config.resolved_data_dir() / "sounds",
        )
        self.llm = llm or LLMClient(self.config.llm,
                                    data_dir=self.config.resolved_data_dir())
        self.commands = command_processor or CommandProcessor(llm_edit=self._llm_edit)
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
        # Serializes engine access between the final pass and the live
        # preview thread (whisper models aren't concurrency-safe).
        self.transcribe_lock = threading.Lock()
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
            # Capture the focused window before anything reacts to the status
            # change: showing the recording pill can shift focus to our own
            # process, and the paste must target the app the user was in.
            self._active_window = self.window_provider.get()
            self._set_status("recording")
            self.state.mode = mode
        self._record_started_at = time.time()
        try:
            self.recorder.start()
        except Exception:
            # Mic unavailable (device vanished, permissions): back to idle so
            # the next attempt isn't silently ignored, and audibly fail.
            with self._lock:
                self._set_status("idle")
            self.sounds.error()
            raise
        self.sounds.start()
        return True

    def stop_recording(self) -> Optional[DictationEvent]:
        with self._lock:
            if self.state.status != "recording":
                return None
            self._set_status("transcribing")
        # Everything below the status change sits in one try/finally: if the
        # recorder, sounds or pipeline raise, the status must still return to
        # idle or the pill stays stuck on "processing" forever.
        try:
            self.sounds.stop()
            audio = self.recorder.stop()
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
        t_stt = time.time()
        with self.transcribe_lock:
            result = self.engine.transcribe(
                audio,
                language=None if language == "auto" else language,
                initial_prompt=self.dictionary.initial_prompt(),
            )
        stt_seconds = time.time() - t_stt
        raw = result.clean_text

        window = self._active_window
        profile = match_profile(window, self.config.profiles)
        overrides = dict(profile.overrides) if profile else {}

        llm_used = False
        llm_seconds = 0.0
        send_after = False
        if mode == "command":
            formatted = raw  # command instructions are used verbatim
        else:
            corrected = self.dictionary.correct(raw)
            if self.config.output.voice_send:
                # strip a trailing "send it" before formatting so neither
                # the rules nor the LLM ever see the command itself
                corrected, send_after = detect_send_command(corrected)
            formatted = format_transcript(corrected, self.config.formatting, overrides)
            # AI cleanup pass (local LLM); rule-based text is the fallback.
            if formatted and self.config.llm.enabled and self.config.llm.format_dictation:
                tone = profile.tone if profile else "auto"
                t_llm = time.time()
                rewritten = self.llm.rewrite(formatted, tone=tone,
                                             app=window.app or window.title,
                                             dictionary=self.dictionary.words)
                llm_seconds = time.time() - t_llm
                if rewritten:
                    formatted = rewritten
                    llm_used = True

        injected = False
        if hasattr(self.injector, "focus_pid"):
            # macOS paste: re-activate the app the user was dictating into in
            # case focus moved (e.g. to our own recording pill) meanwhile.
            self.injector.focus_pid = window.pid
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

        auto_sent = False
        if send_after and hasattr(self.injector, "press_return") \
                and (injected or not formatted):
            # "send it" alone (no other words) sends whatever is already
            # typed in the target field - intentional.
            try:
                time.sleep(0.05)
                self.injector.press_return()
                auto_sent = True
                self.state.session_text = ""  # message sent; context over
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
            llm_used=llm_used,
            stt_seconds=stt_seconds,
            llm_seconds=llm_seconds,
            auto_sent=auto_sent,
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

    def _llm_edit(self, instruction: str, text: str) -> Optional[str]:
        """Command-mode hook: free-form edits via the local LLM."""
        if not (self.config.llm.enabled and self.config.llm.command_mode):
            return None
        return self.llm.edit(instruction, text)

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
        if hasattr(self.injector, "focus_pid"):
            self.injector.focus_pid = self._active_window.pid
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
        self.recorder.close()
        if self._engine is not None:
            self._engine.close()
