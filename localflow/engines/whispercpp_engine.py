"""whisper.cpp backend via pywhispercpp - light, fast, pure-CPU friendly."""

from __future__ import annotations

import contextlib
import os
import sys
from typing import Iterator, Optional

import numpy as np

from .base import Segment, STTEngine, TranscriptionResult


@contextlib.contextmanager
def _quiet_stderr() -> Iterator[None]:
    """Silence C-level stderr (whisper.cpp logs straight to fd 2)."""
    try:
        sys.stderr.flush()
        fd = sys.stderr.fileno()
    except (AttributeError, OSError, ValueError):
        yield
        return
    saved = os.dup(fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, fd)
        yield
    finally:
        os.dup2(saved, fd)
        os.close(saved)
        os.close(devnull)


class WhisperCppEngine(STTEngine):
    name = "whisper.cpp"

    def __init__(
        self,
        model_path: str,
        threads: int = 4,
        beam_size: int = 1,
    ) -> None:
        from pywhispercpp.model import Model  # lazy import

        self.model_path = model_path
        self.threads = threads
        self.beam_size = beam_size
        with _quiet_stderr():
            self._model = Model(
                model_path,
                n_threads=threads,
                print_progress=False,
                print_realtime=False,
                redirect_whispercpp_logs_to=None,  # None = devnull in pywhispercpp
            )

    def transcribe(
        self,
        audio: np.ndarray,
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None,
    ) -> TranscriptionResult:
        if audio.size == 0:
            return TranscriptionResult(text="")
        kwargs = {}
        if language and language != "auto":
            kwargs["language"] = language
        elif self._is_multilingual():
            kwargs["language"] = "auto"
        if initial_prompt:
            kwargs["initial_prompt"] = initial_prompt


        audio = np.ascontiguousarray(audio, dtype=np.float32)
        with _quiet_stderr():
            raw_segments = self._model.transcribe(audio, **kwargs)

        segments = [
            # pywhispercpp reports times in centiseconds
            Segment(start=s.t0 / 100.0, end=s.t1 / 100.0, text=s.text)
            for s in raw_segments
        ]
        text = " ".join(s.text.strip() for s in segments).strip()
        detected = None
        if language and language != "auto":
            detected = language
        return TranscriptionResult(
            text=text,
            language=detected,
            duration=audio.shape[0] / 16000.0,
            segments=segments,
        )

    def _is_multilingual(self) -> bool:
        # English-only ggml models are conventionally named *.en*
        return ".en" not in str(self.model_path)
