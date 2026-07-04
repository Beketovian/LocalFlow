"""End-to-end tests with a REAL Whisper model.

Speech is synthesized with espeak-ng, transcribed by whisper.cpp (ggml tiny),
formatted, and injected - the entire production pipeline with zero mocks.

The tiny model + robotic TTS make word-perfect output unrealistic, so
assertions are fuzzy: enough keywords must survive the round trip.
"""

from __future__ import annotations

import difflib
import re

import pytest

from localflow.app import FlowController
from localflow.audio import ArrayRecorder
from localflow.config import Config
from localflow.history import History
from localflow.injector import CallbackInjector
from localflow.sounds import SoundPlayer

pytestmark = pytest.mark.integration


def similarity(a: str, b: str) -> float:
    norm = lambda s: re.sub(r"[^a-z0-9 ]", "", s.lower())  # noqa: E731
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()


class TestRealTranscription:
    def test_simple_sentence(self, whisper_engine, speech):
        audio = speech("Hello world, this is a test.")
        result = whisper_engine.transcribe(audio)
        assert similarity(result.clean_text, "Hello world, this is a test.") > 0.6

    def test_longer_dictation(self, whisper_engine, speech):
        text = "The quick brown fox jumps over the lazy dog."
        result = whisper_engine.transcribe(speech(text, speed=130))
        assert similarity(result.clean_text, text) > 0.6

    def test_empty_audio(self, whisper_engine):
        import numpy as np

        result = whisper_engine.transcribe(np.zeros(0, dtype=np.float32))
        assert result.text == ""

    def test_segments_have_timestamps(self, whisper_engine, speech):
        result = whisper_engine.transcribe(speech("One two three four five six."))
        assert result.segments
        assert result.segments[-1].end > result.segments[0].start


class TestFullPipeline:
    def _controller(self, engine, audio):
        config = Config()
        config.save_history = False
        config.audio.feedback_sounds = False
        return FlowController(
            config=config,
            engine=engine,
            recorder=ArrayRecorder(audio),
            injector=CallbackInjector(),
            history=History(":memory:"),
            sounds=SoundPlayer(enabled=False),
        )

    def test_dictation_end_to_end(self, whisper_engine, speech):
        audio = speech("Hello world, this is a test.")
        controller = self._controller(whisper_engine, audio)
        assert controller.start_recording()
        event = controller.stop_recording()
        assert event is not None and event.injected
        assert similarity(event.formatted_text, "Hello world, this is a test.") > 0.55
        # history captured it
        assert controller.history.recent()[0].formatted_text == event.formatted_text

    def test_quiet_speech_normalization(self, whisper_engine, speech):
        # Simulate whispering: scale way down; RMS normalization must rescue it
        audio = speech("Testing quiet voice input.") * 0.02
        controller = self._controller(whisper_engine, audio)
        controller.start_recording()
        event = controller.stop_recording()
        assert event.formatted_text, "quiet audio should still transcribe"

    def test_transcribe_realtime_factor(self, whisper_engine, speech):
        # tiny on CPU should be far faster than real time; catch regressions
        audio = speech("Measuring transcription speed with a moderately long sentence.")
        import time

        t0 = time.time()
        whisper_engine.transcribe(audio)
        elapsed = time.time() - t0
        audio_seconds = audio.shape[0] / 16000
        assert elapsed < audio_seconds * 3, f"too slow: {elapsed:.1f}s for {audio_seconds:.1f}s audio"
