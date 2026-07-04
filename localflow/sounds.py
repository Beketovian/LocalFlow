"""Audio feedback - soft tones on record start/stop/error."""

from __future__ import annotations

import numpy as np

from .audio import SAMPLE_RATE


def tone(freq: float, duration: float = 0.09, volume: float = 0.18,
         rate: int = SAMPLE_RATE) -> np.ndarray:
    t = np.linspace(0, duration, int(rate * duration), endpoint=False)
    wave = np.sin(2 * np.pi * freq * t)
    # quick fade in/out to avoid clicks
    fade = min(len(t) // 6, int(rate * 0.01)) or 1
    envelope = np.ones_like(wave)
    envelope[:fade] = np.linspace(0, 1, fade)
    envelope[-fade:] = np.linspace(1, 0, fade)
    return (wave * envelope * volume).astype(np.float32)


START_SOUND = np.concatenate([tone(660), tone(880)])
STOP_SOUND = np.concatenate([tone(880), tone(660)])
ERROR_SOUND = np.concatenate([tone(220, 0.12), tone(196, 0.16)])


class SoundPlayer:
    """Plays feedback tones; silently does nothing when audio out is missing."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def play(self, samples: np.ndarray) -> None:
        if not self.enabled:
            return
        try:
            import sounddevice as sd

            sd.play(samples, SAMPLE_RATE, blocking=False)
        except Exception:
            pass

    def start(self) -> None:
        self.play(START_SOUND)

    def stop(self) -> None:
        self.play(STOP_SOUND)

    def error(self) -> None:
        self.play(ERROR_SOUND)
