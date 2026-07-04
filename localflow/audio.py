"""Audio capture and processing.

All audio flows through the app as mono float32 numpy arrays at 16 kHz (what
Whisper expects). Microphone capture uses `sounddevice` when available; every
consumer depends only on the small `Recorder` interface so tests (and headless
environments) can substitute `ArrayRecorder`.
"""

from __future__ import annotations

import threading
import wave
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

SAMPLE_RATE = 16000


# ------------------------------------------------------------------ wav helpers


def load_wav(path: str | Path, target_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Load a WAV file as mono float32 in [-1, 1], resampled to target_rate."""
    with wave.open(str(path), "rb") as wf:
        rate = wf.getframerate()
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())

    if width == 2:
        data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif width == 4:
        data = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif width == 1:
        data = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {width}")

    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    if rate != target_rate:
        data = resample(data, rate, target_rate)
    return data


def save_wav(path: str | Path, audio: np.ndarray, rate: int = SAMPLE_RATE) -> None:
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm.tobytes())


def resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Linear-interpolation resampler - adequate for speech into Whisper."""
    if src_rate == dst_rate or audio.size == 0:
        return audio.astype(np.float32)
    duration = audio.shape[0] / src_rate
    dst_len = max(1, int(round(duration * dst_rate)))
    src_x = np.linspace(0.0, 1.0, num=audio.shape[0], endpoint=False)
    dst_x = np.linspace(0.0, 1.0, num=dst_len, endpoint=False)
    return np.interp(dst_x, src_x, audio).astype(np.float32)


def rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


def normalize_rms(audio: np.ndarray, target_rms: float = 0.06, max_gain: float = 30.0) -> np.ndarray:
    """Boost quiet audio (e.g. whispering) toward a target RMS level.

    This is what lets barely-audible speech transcribe well - Wispr Flow's
    'whisper mode'. Gain is capped so silence isn't amplified into noise.
    """
    level = rms(audio)
    if level < 1e-6:
        return audio
    gain = min(target_rms / level, max_gain)
    if gain <= 1.0:
        return audio
    return np.clip(audio * gain, -1.0, 1.0).astype(np.float32)


# -------------------------------------------------------------------- recorders


class Recorder:
    """Interface: start capturing, stop and get the audio back."""

    def start(self) -> None:
        raise NotImplementedError

    def stop(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def is_recording(self) -> bool:
        raise NotImplementedError

    def snapshot(self) -> np.ndarray:
        """Audio captured so far without stopping (for streaming preview)."""
        raise NotImplementedError


class ArrayRecorder(Recorder):
    """Deterministic recorder for tests/headless use: 'records' preset audio."""

    def __init__(self, audio: Optional[np.ndarray] = None) -> None:
        self.audio = audio if audio is not None else np.zeros(0, dtype=np.float32)
        self._recording = False

    def start(self) -> None:
        self._recording = True

    def stop(self) -> np.ndarray:
        self._recording = False
        return self.audio

    @property
    def is_recording(self) -> bool:
        return self._recording

    def snapshot(self) -> np.ndarray:
        return self.audio


class MicrophoneRecorder(Recorder):
    """Real microphone capture via sounddevice (imported lazily)."""

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        device: Optional[str] = None,
        max_seconds: float = 120.0,
        on_chunk: Optional[Callable[[np.ndarray], None]] = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.device = device
        self.max_seconds = max_seconds
        self.on_chunk = on_chunk
        self._chunks: List[np.ndarray] = []
        self._lock = threading.Lock()
        self._stream = None

    def start(self) -> None:
        import sounddevice as sd  # lazy: needs PortAudio

        with self._lock:
            self._chunks = []

        def callback(indata, frames, time_info, status) -> None:
            mono = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()
            with self._lock:
                self._chunks.append(mono)
                total = sum(c.shape[0] for c in self._chunks)
            if self.on_chunk is not None:
                self.on_chunk(mono)
            if total > self.max_seconds * self.sample_rate:
                raise sd.CallbackStop()

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            device=self.device,
            callback=callback,
        )
        self._stream.start()

    def stop(self) -> np.ndarray:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        with self._lock:
            if not self._chunks:
                return np.zeros(0, dtype=np.float32)
            return np.concatenate(self._chunks).astype(np.float32)

    @property
    def is_recording(self) -> bool:
        return self._stream is not None

    def snapshot(self) -> np.ndarray:
        with self._lock:
            if not self._chunks:
                return np.zeros(0, dtype=np.float32)
            return np.concatenate(self._chunks).astype(np.float32)
