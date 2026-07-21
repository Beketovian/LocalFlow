"""faster-whisper (CTranslate2) backend - best accuracy/speed on CPU and GPU."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base import Segment, STTEngine, TranscriptionResult


class FasterWhisperEngine(STTEngine):
    name = "faster-whisper"

    def __init__(
        self,
        model: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        threads: int = 4,
        beam_size: int = 1,
    ) -> None:
        from faster_whisper import WhisperModel  # lazy import

        self.beam_size = beam_size
        self._model = WhisperModel(
            model,
            device=device,
            compute_type=compute_type,
            cpu_threads=threads,
        )

    def transcribe(
        self,
        audio: np.ndarray,
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None,
    ) -> TranscriptionResult:
        if audio.size == 0:
            return TranscriptionResult(text="")
        lang = None if language in (None, "auto") else language
        raw_segments, info = self._model.transcribe(
            np.ascontiguousarray(audio, dtype=np.float32),
            language=lang,
            initial_prompt=initial_prompt,
            beam_size=max(1, self.beam_size),
            vad_filter=True,
            condition_on_previous_text=True,
        )
        
        segments = []
        for s in raw_segments:
            if getattr(s, "no_speech_prob", 0.0) > 0.5:
                continue
            if getattr(s, "compression_ratio", 0.0) > 2.4:
                continue
            segments.append(Segment(start=s.start, end=s.end, text=s.text))
        text = " ".join(s.text.strip() for s in segments).strip()
        return TranscriptionResult(
            text=text,
            language=getattr(info, "language", lang),
            duration=audio.shape[0] / 16000.0,
            segments=segments,
        )
