"""Audio feedback - soft tones on record start/stop/error.

On macOS the tones go through NSSound (the system audio server) instead of
PortAudio: sd.play() opens and closes an output stream around every beep,
which pops, and the stop tone plays exactly while Whisper saturates the
CPU/GPU - PortAudio underruns audibly crackle, coreaudiod doesn't care.
"""

from __future__ import annotations

import io
import sys
import wave

import numpy as np

from .audio import SAMPLE_RATE

_PLAYBACK_RATE = 48000  # native output rate; no device resampling clicks


def tone(freq: float, duration: float = 0.09, volume: float = 0.18,
         rate: int = _PLAYBACK_RATE) -> np.ndarray:
    t = np.linspace(0, duration, int(rate * duration), endpoint=False)
    wave_ = np.sin(2 * np.pi * freq * t)
    # raised-cosine fade in/out to avoid clicks
    fade = min(len(t) // 4, int(rate * 0.015)) or 1
    envelope = np.ones_like(wave_)
    ramp = (1 - np.cos(np.linspace(0, np.pi, fade))) / 2
    envelope[:fade] = ramp
    envelope[-fade:] = ramp[::-1]
    return (wave_ * envelope * volume).astype(np.float32)


START_SOUND = np.concatenate([tone(660), tone(880)])
STOP_SOUND = np.concatenate([tone(880), tone(660)])
ERROR_SOUND = np.concatenate([tone(220, 0.12), tone(196, 0.16)])


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

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._nssounds: dict = {}  # id(samples) -> NSSound, pre-rendered

    def _play_nssound(self, samples: np.ndarray) -> bool:
        try:
            from AppKit import NSSound
            from Foundation import NSData

            sound = self._nssounds.get(id(samples))
            if sound is None:
                data = NSData.dataWithBytes_length_(
                    _wav_bytes(samples), len(_wav_bytes(samples)))
                sound = NSSound.alloc().initWithData_(data)
                if sound is None:
                    return False
                self._nssounds[id(samples)] = sound
            if sound.isPlaying():
                sound.stop()
            sound.play()
            return True
        except Exception:
            return False

    def play(self, samples: np.ndarray) -> None:
        if not self.enabled:
            return
        if sys.platform == "darwin" and self._play_nssound(samples):
            return
        try:
            import sounddevice as sd

            sd.play(samples, _PLAYBACK_RATE, blocking=False)
        except Exception:
            pass

    def start(self) -> None:
        self.play(START_SOUND)

    def stop(self) -> None:
        self.play(STOP_SOUND)

    def error(self) -> None:
        self.play(ERROR_SOUND)
