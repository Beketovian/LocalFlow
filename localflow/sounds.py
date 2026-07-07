"""Audio feedback - soft tones on record start/stop/error.

Two design rules learned the hard way:

* Playback goes through NSSound on macOS (the system audio server), never
  PortAudio: sd.play() opens/closes an output stream around every beep
  (audible pop) and underruns when Whisper saturates the machine right
  after the hotkey is released.
* The tones are synthesized plucks (fundamental + decaying overtones, like
  a soft marimba), not raw sine sweeps - much closer to Wispr Flow's
  feedback sound. Drop your own start.wav / stop.wav / error.wav (or
  .aiff/.mp3 on macOS) into <data dir>/sounds/ to replace them entirely.
"""

from __future__ import annotations

import io
import sys
import wave
from pathlib import Path
from typing import Optional

import numpy as np

_PLAYBACK_RATE = 48000  # native output rate; no device resampling clicks


def _pluck(freq: float, duration: float = 0.18, volume: float = 0.22,
           delay: float = 0.0, decay: float = 22.0,
           rate: int = _PLAYBACK_RATE) -> np.ndarray:
    """A soft struck-note: fundamental plus fast-decaying overtones."""
    n = int(rate * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    tone = (
        np.sin(2 * np.pi * freq * t)
        + 0.35 * np.sin(2 * np.pi * 2 * freq * t) * np.exp(-t * decay * 1.8)
        + 0.12 * np.sin(2 * np.pi * 3 * freq * t) * np.exp(-t * decay * 2.6)
    )
    envelope = np.exp(-t * decay)
    attack = max(1, int(rate * 0.004))  # declick the onset
    envelope[:attack] *= np.linspace(0, 1, attack)
    out = (tone * envelope * volume).astype(np.float32)
    if delay > 0:
        out = np.concatenate([np.zeros(int(rate * delay), dtype=np.float32), out])
    return out


def _mix(*parts: np.ndarray) -> np.ndarray:
    length = max(p.shape[0] for p in parts)
    out = np.zeros(length, dtype=np.float32)
    for p in parts:
        out[: p.shape[0]] += p
    return np.clip(out, -1.0, 1.0)


# C5->G5 "ready" rise; G5->C5 "done" fall; low dissonant pair for errors.
START_SOUND = _mix(_pluck(523.25), _pluck(783.99, delay=0.07))
STOP_SOUND = _mix(_pluck(783.99), _pluck(523.25, delay=0.07))
ERROR_SOUND = _mix(_pluck(220.0, duration=0.28, decay=14.0),
                   _pluck(207.65, duration=0.28, decay=14.0, delay=0.09))

_CUSTOM_STEMS = {"start": None, "stop": None, "error": None}
_CUSTOM_EXTS = (".wav", ".aiff", ".aif", ".mp3", ".m4a", ".caf")


def _wav_bytes(samples: np.ndarray, rate: int = _PLAYBACK_RATE) -> bytes:
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


class SoundPlayer:
    """Plays feedback tones; silently does nothing when audio out is missing."""

    def __init__(self, enabled: bool = True,
                 sounds_dir: Optional[Path] = None) -> None:
        self.enabled = enabled
        self.sounds_dir = Path(sounds_dir) if sounds_dir else None
        self._nssounds: dict = {}  # cache: name -> NSSound

    def _custom_file(self, name: str) -> Optional[Path]:
        if self.sounds_dir is None:
            return None
        for ext in _CUSTOM_EXTS:
            p = self.sounds_dir / f"{name}{ext}"
            if p.exists():
                return p
        return None

    def _nssound_for(self, name: str, samples: np.ndarray):
        from AppKit import NSSound
        from Foundation import NSData

        sound = self._nssounds.get(name)
        if sound is not None:
            return sound
        custom = self._custom_file(name)
        if custom is not None:
            sound = NSSound.alloc().initWithContentsOfFile_byReference_(
                str(custom), True)
        if sound is None:
            data = _wav_bytes(samples)
            sound = NSSound.alloc().initWithData_(
                NSData.dataWithBytes_length_(data, len(data)))
        if sound is not None:
            self._nssounds[name] = sound
        return sound

    def play(self, samples: np.ndarray, name: str = "custom") -> None:
        if not self.enabled:
            return
        if sys.platform == "darwin":
            try:
                sound = self._nssound_for(name, samples)
                if sound is not None:
                    if sound.isPlaying():
                        sound.stop()
                    sound.play()
                    return
            except Exception:
                pass
        try:
            import sounddevice as sd

            sd.play(samples, _PLAYBACK_RATE, blocking=False)
        except Exception:
            pass

    def start(self) -> None:
        self.play(START_SOUND, "start")

    def stop(self) -> None:
        self.play(STOP_SOUND, "stop")

    def error(self) -> None:
        self.play(ERROR_SOUND, "error")
