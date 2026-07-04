"""Speech-to-text engine interface.

Engines take 16 kHz mono float32 audio and return a TranscriptionResult.
Everything above this layer (formatting, injection, history, UI) is engine
agnostic, which is what makes the app testable without a model and lets users
swap faster-whisper <-> whisper.cpp freely.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    text: str
    language: Optional[str] = None
    duration: float = 0.0
    segments: List[Segment] = field(default_factory=list)

    @property
    def clean_text(self) -> str:
        return self.text.strip()


class STTEngine:
    name: str = "base"

    def transcribe(
        self,
        audio: np.ndarray,
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None,
    ) -> TranscriptionResult:
        raise NotImplementedError

    def close(self) -> None:  # optional resource cleanup
        pass
