"""Live transcription preview while recording.

Whisper isn't natively streaming, so LocalFlow does what most local dictation
tools do: periodically re-transcribe the audio captured so far in a background
thread and emit partial text. The final (authoritative) pass happens on stop.
"""

from __future__ import annotations

import threading
from typing import Callable, Optional

from .audio import Recorder
from .engines.base import STTEngine


class StreamingPreview:
    def __init__(
        self,
        engine: STTEngine,
        recorder: Recorder,
        on_partial: Callable[[str], None],
        interval: float = 1.5,
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None,
    ) -> None:
        self.engine = engine
        self.recorder = recorder
        self.on_partial = on_partial
        self.interval = interval
        self.language = language
        self.initial_prompt = initial_prompt
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.last_partial: str = ""

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            audio = self.recorder.snapshot()
            if audio.size < 8000:  # <0.5s - nothing useful yet
                continue
            try:
                result = self.engine.transcribe(
                    audio, language=self.language, initial_prompt=self.initial_prompt
                )
            except Exception:
                continue
            text = result.clean_text
            if text and text != self.last_partial:
                self.last_partial = text
                self.on_partial(text)
