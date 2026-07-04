import numpy as np
import pytest

from localflow.audio import (
    ArrayRecorder,
    load_wav,
    normalize_rms,
    resample,
    rms,
    save_wav,
)
from localflow.vad import SilenceDetector, trim_silence


def sine(freq=440.0, seconds=1.0, rate=16000, amp=0.5):
    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class TestWavIO:
    def test_roundtrip(self, tmp_path):
        audio = sine()
        path = tmp_path / "t.wav"
        save_wav(path, audio)
        loaded = load_wav(path)
        assert loaded.shape[0] == audio.shape[0]
        assert np.abs(loaded - audio).max() < 0.01

    def test_resample_on_load(self, tmp_path):
        audio = sine(rate=8000, seconds=1.0)
        path = tmp_path / "t8.wav"
        save_wav(path, audio, rate=8000)
        loaded = load_wav(path, target_rate=16000)
        assert abs(loaded.shape[0] - 16000) < 20

    def test_resample_identity(self):
        audio = sine()
        assert resample(audio, 16000, 16000) is audio or \
            np.array_equal(resample(audio, 16000, 16000), audio)


class TestNormalize:
    def test_boosts_quiet_audio(self):
        quiet = sine(amp=0.005)
        boosted = normalize_rms(quiet, target_rms=0.06)
        assert rms(boosted) > rms(quiet) * 3

    def test_leaves_loud_audio(self):
        loud = sine(amp=0.5)
        assert np.array_equal(normalize_rms(loud, target_rms=0.06), loud)

    def test_silence_not_amplified_to_noise(self):
        silence = np.zeros(16000, dtype=np.float32)
        out = normalize_rms(silence)
        assert rms(out) == 0.0


class TestSilenceDetector:
    def test_stops_after_trailing_silence(self):
        det = SilenceDetector(threshold=0.01, stop_after_seconds=0.5)
        speech = sine(seconds=1.0, amp=0.3)
        silence = np.zeros(16000, dtype=np.float32)
        assert det.feed(speech) is False
        assert det.feed(silence) is True

    def test_no_stop_before_any_speech(self):
        det = SilenceDetector(threshold=0.01, stop_after_seconds=0.3)
        silence = np.zeros(32000, dtype=np.float32)
        assert det.feed(silence) is False
        assert det.heard_speech is False

    def test_reset(self):
        det = SilenceDetector(threshold=0.01, stop_after_seconds=0.2)
        det.feed(sine(seconds=0.5, amp=0.3))
        det.feed(np.zeros(16000, dtype=np.float32))
        det.reset()
        assert det.heard_speech is False


class TestTrimSilence:
    def test_trims_padding(self):
        pad = np.zeros(16000, dtype=np.float32)
        audio = np.concatenate([pad, sine(seconds=1.0, amp=0.3), pad])
        trimmed = trim_silence(audio)
        assert trimmed.shape[0] < audio.shape[0]
        assert trimmed.shape[0] >= 16000  # speech kept

    def test_all_silence_returned_as_is(self):
        silence = np.zeros(8000, dtype=np.float32)
        assert trim_silence(silence).shape[0] == 8000


class TestArrayRecorder:
    def test_record_cycle(self):
        rec = ArrayRecorder(sine())
        assert rec.is_recording is False
        rec.start()
        assert rec.is_recording is True
        assert rec.snapshot().shape[0] == 16000
        audio = rec.stop()
        assert audio.shape[0] == 16000
        assert rec.is_recording is False
