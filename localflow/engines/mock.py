"""Deterministic engine for tests and dry runs."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .base import Segment, STTEngine, TranscriptionResult


class MockEngine(STTEngine):
    """Returns canned transcripts; records what it was asked to transcribe."""

    name = "mock"

    def __init__(self, responses: Optional[List[str]] = None) -> None:
        self.responses = list(responses or ["hello world"])
        self.calls: List[dict] = []
        self._i = 0

    def transcribe(
        self,
        audio: np.ndarray,
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None,
    ) -> TranscriptionResult:
        self.calls.append(
            {
                "samples": int(audio.shape[0]) if audio is not None else 0,
                "language": language,
                "initial_prompt": initial_prompt,
            }
        )
        text = self.responses[min(self._i, len(self.responses) - 1)]
        self._i += 1
        duration = (audio.shape[0] / 16000.0) if audio is not None and audio.size else 0.0
        return TranscriptionResult(
            text=text,
            language=language if language not in (None, "auto") else "en",
            duration=duration,
            segments=[Segment(0.0, duration, text)],
        )
