"""Speech-to-text engine interface.

Engines take 16 kHz mono float32 audio and return a TranscriptionResult.
Everything above this layer (formatting, injection, history, UI) is engine
agnostic, which is what makes the app testable without a model and lets users
swap faster-whisper <-> whisper.cpp freely.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class Segment:
    start: float
    end: float
    text: str


# Whisper labels non-speech instead of staying quiet: "[BLANK_AUDIO]",
# "[MUSIC]", "*coughs*", "(keyboard clacking)", ♪ lyrics ♪. Nothing a user
# dictates comes back in square brackets/asterisks/music notes, so those are
# stripped wholesale; parentheses can be legitimate, so only known noise
# words are removed there.
_BRACKET_NOISE = re.compile(r"\[[^\]]*\]|\*[^*\n]*\*|♪[^♪\n]*♪|♪")
_PAREN_NOISE = re.compile(
    r"\(\s*[a-z\s]*\b(?:silence|silent|music|applause|laugh\w*|cough\w*|"
    r"typing|click\w*|clack\w*|breath\w*|sigh\w*|noise|static|inaudible|"
    r"blank|beep\w*|wind|birds?|chirp\w*|speaking|foreign language|"
    r"no (?:audio|speech))\b[a-z\s]*\)",
    re.IGNORECASE,
)


@dataclass
class TranscriptionResult:
    text: str
    language: Optional[str] = None
    duration: float = 0.0
    segments: List[Segment] = field(default_factory=list)

    @property
    def clean_text(self) -> str:
        text = _BRACKET_NOISE.sub(" ", self.text)
        text = _PAREN_NOISE.sub(" ", text)
        return re.sub(r"\s{2,}", " ", text).strip()


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
