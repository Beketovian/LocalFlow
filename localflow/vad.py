"""Silence detection for hands-free dictation.

A deliberately simple energy-based voice activity detector: hands-free mode
stops recording once trailing silence exceeds a threshold. (Whisper itself is
robust to leading/trailing silence; this only decides *when to stop*.)
"""

from __future__ import annotations

import numpy as np

from .audio import SAMPLE_RATE, rms


class SilenceDetector:
    def __init__(
        self,
        threshold: float = 0.012,
        stop_after_seconds: float = 1.6,
        sample_rate: int = SAMPLE_RATE,
        frame_ms: int = 30,
    ) -> None:
        self.threshold = threshold
        self.stop_after_seconds = stop_after_seconds
        self.sample_rate = sample_rate
        self.frame_len = int(sample_rate * frame_ms / 1000)
        self._silence_frames = 0
        self._voiced_once = False

    def reset(self) -> None:
        self._silence_frames = 0
        self._voiced_once = False

    @property
    def heard_speech(self) -> bool:
        return self._voiced_once

    def feed(self, chunk: np.ndarray) -> bool:
        """Feed captured audio; returns True when it's time to stop."""
        if chunk.size == 0:
            return False
        for start in range(0, chunk.shape[0], self.frame_len):
            frame = chunk[start : start + self.frame_len]
            if rms(frame) >= self.threshold:
                self._voiced_once = True
                self._silence_frames = 0
            else:
                self._silence_frames += 1
        if not self._voiced_once:
            return False
        silent_seconds = self._silence_frames * self.frame_len / self.sample_rate
        return silent_seconds >= self.stop_after_seconds

    def trailing_silence_seconds(self) -> float:
        return self._silence_frames * self.frame_len / self.sample_rate


def trim_silence(
    audio: np.ndarray,
    threshold: float = 0.008,
    sample_rate: int = SAMPLE_RATE,
    pad_ms: int = 150,
) -> np.ndarray:
    """Trim leading/trailing silence, keeping a small pad for natural onsets."""
    if audio.size == 0:
        return audio
    frame = int(sample_rate * 0.02)
    n_frames = max(1, audio.shape[0] // frame)
    energies = [rms(audio[i * frame : (i + 1) * frame]) for i in range(n_frames)]
    voiced = [i for i, e in enumerate(energies) if e >= threshold]
    if not voiced:
        return audio
    pad = int(sample_rate * pad_ms / 1000)
    start = max(0, voiced[0] * frame - pad)
    end = min(audio.shape[0], (voiced[-1] + 1) * frame + pad)
    return audio[start:end]
